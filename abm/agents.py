from __future__ import annotations

import numpy as np
from utils import sample_categorical, softmax
from array_to_df import using_multiindex
from dataclasses import dataclass
from pydantic import BaseModel
import os

class AgentConfig(BaseModel):
    beta_teacher: float
    beta_evo: float
    agent_types: list[str]
    T: list[int]
    d: list[int]
    eps: list[float]
    phi: list[float]
    learn_d: bool
    N_0: list[int]
    N_o: list[int]


@dataclass
class AgentPop:
    """
    Agent population state and action logic (vectorized).

    Stored arrays (all include the generation axis):
    - T_i: lifetime trials per agent                                                 # [R,G,N]
    - d_i: exploration depth (max length / breadth of exploration)                    # [R,G,N]
    - eps_i: exploration probability                                                  # [R,G,N]
    - K: repertoire (boolean array indicating which strategies are known)             # [R,G,N,X]
    - perf: performance accumulator (sum of rewards)                                   # [R,G,N]
    """

    R: int # number of replications
    G: int # number of generations
    N: int # number of agents
    P: int # number of tasks
    L: int # maximum strategy length

    agent_config: AgentConfig
    rng: np.random.Generator

    g: int = 0 # current generation index
    t: int = 0 # current trial index

    X: int | None = None # number of all possible strategies
    T_max: int | None = None # maximum lifetime trials

    T_i: np.ndarray | None = None # [R,G,N] lifetime trials per agent
    d_i: np.ndarray | None = None # [R,G,N] exploration depth (max length for novel prefixes)
    eps_i: np.ndarray | None = None # [R,G,N] exploration probability
    phi_i: np.ndarray | None = None # [R,G,N] exploration probability
    AT: np.ndarray | None = None # [R,G,N] agent type indices

    K: np.ndarray | None = None # [R,G,N,X] repertoire (known strategies and their payoffs)
    perf: np.ndarray | None = None # [R,G,N] performance accumulator
    x: np.ndarray | None = None # [R,G,T_max,N] strategy indices at time t
    m: np.ndarray | None = None # [R,G,T_max,N] meta strategy indices at time t
    r: np.ndarray | None = None # [R,G,T_max,N] reward at time t

    #utils
    pow2: np.ndarray | None = None # [L+1] 2^l for l<=L

    def __post_init__(self) -> None:
        assert (
            len(self.agent_config.agent_types)
            == len(self.agent_config.T)
            == len(self.agent_config.d)
            == len(self.agent_config.eps)
            == len(self.agent_config.N_0)
            == len(self.agent_config.N_o)
        )
        assert int(sum(self.agent_config.N_o)) == self.N == int(sum(self.agent_config.N_0))
        self.X = 2 ** self.L
        
        # initialize agent type indices for each generation
        AT = np.zeros((self.R, self.G, self.N), dtype=np.int32)
        at_0 = np.concatenate(
            [np.full(int(n), i, dtype=np.int32) for i, n in enumerate(self.agent_config.N_0)],
            axis=0,
        )
        at_o = np.concatenate(
            [np.full(int(n), i, dtype=np.int32) for i, n in enumerate(self.agent_config.N_o)],
            axis=0,
        )
        AT[:, 0, :] = at_0[None, :]
        if self.G > 1:
            AT[:, 1:, :] = at_o[None, None, :]
        self.AT = AT

        self.T_i = np.array(self.agent_config.T)[AT]
        self.T_max = np.max(self.T_i)
        self.d_i = np.array(self.agent_config.d)[AT]
        self.eps_i = np.array(self.agent_config.eps)[AT]
        self.phi_i = np.array(self.agent_config.phi)[AT]
        
        self.K = np.zeros((self.R, self.G, self.N, self.X), dtype=bool)
        self.perf = np.zeros((self.R, self.G, self.N), dtype=float)
        self.pow2 = np.power(2, np.arange(self.L + 1))

        self.x = np.full((self.R, self.G, self.T_max, self.N), -1, dtype=np.int32)
        self.m = np.full((self.R, self.G, self.T_max, self.N), -1, dtype=np.int32)
        self.r = np.full((self.R, self.G, self.T_max, self.N), 0.0, dtype=float)

    def is_alive(self) -> np.ndarray:
        """
        Returns alive mask for this trial.
        alive: (R,N) bool                                                           # [R,N]
        """
        assert self.T_i is not None
        return (self.t < self.T_i[:, self.g, :])                                   # [R,N]


    def explore(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Samples a candidate strategy index for exploration.

        Note: in this vectorized prototype, exploration depth controls the *breadth* of the
        strategy index range searched: max_idx = min(2^d_i, X).

        Returns:
            s_idx: (R,N) int32                                                      # [R,N]
        """
        assert self.d_i is not None and self.K is not None and self.pow2 is not None

        max_idx = np.minimum(self.pow2[self.d_i[:, self.g, :]], self.X).astype(np.int32)  # [R,N]
        max_idx = np.maximum(max_idx, 1)  # ensure valid upper bound
        x_idx = self.rng.integers(0, max_idx, size=(self.R, self.N), dtype=np.int32)      # [R,N]
        return x_idx

    def exploit(self) -> np.ndarray:
        """Sample a known strategy index for each (r, n) from K[:, g, n, :]."""
        K = self.K[:, self.g].astype(float)          # (R, N, X)

        row_sum = K.sum(axis=-1, keepdims=True)      # (R, N, 1)
        valid_mask = (row_sum > 0).squeeze(axis=-1)   # (R, N)

        x_idx = np.full((self.R, self.N), -1, dtype=np.int32)
        K_valid = K[valid_mask]                   # (M, X) where M is number of valid (r,n) pairs
        row_sum_valid = row_sum[valid_mask]       # (M, 1)
        probs = K_valid / row_sum_valid           # (M, X)

        # Draw one categorical sample per valid (r,n)
        one_hot = self.rng.multinomial(1, probs) # (M, X)
        x_idx_valid = one_hot.argmax(axis=-1)    # (M,)
        x_idx[valid_mask] = x_idx_valid
        
        return x_idx

    def act(self) -> np.ndarray:
        """
        Returns strategy indices for each agent.
        s_idx: (R,N) int32                                                          # [R,N]
        """
        # With probability (1-eps), select safe prefix; otherwise explore/exploit with probability eps
        m = np.zeros((self.R, self.N), dtype=np.int32)
        x_idx = np.zeros((self.R, self.N), dtype=np.int32)

        explore_idx = self.explore() # [R,N]
        exploit_idx = self.exploit() # [R,N]

        go_safe = (self.rng.random(size=(self.R, self.N)) < (1 - self.eps_i[:, self.g, :]))
        go_exploit = (self.rng.random(size=(self.R, self.N)) < self.phi_i[:, self.g, :])
        go_exploit = go_exploit & (exploit_idx != -1) & ~go_safe
        go_explore = ~go_safe & ~go_exploit

        m = np.where(go_exploit, 1, m)
        m = np.where(go_explore, 2, m)

        x_idx = np.where(go_exploit, exploit_idx, x_idx)
        x_idx = np.where(go_explore, explore_idx, x_idx)

        self.x[:, self.g, self.t, :] = x_idx
        self.m[:, self.g, self.t, :] = m
        return x_idx

    def update(self, alive: np.ndarray, x_idx: np.ndarray, reward: np.ndarray) -> None:
        """
        Updates within-lifetime state for generation g given observed rewards.

        - Updates repertoire values K for the chosen strategy.
        - Accumulates performance perf.
        """
        assert self.K is not None and self.perf is not None

        r_idx = np.arange(self.R)[:, None]
        n_idx = np.arange(self.N)[None, :]

        discovered = (reward > 0) & alive
        self.K[r_idx, self.g, n_idx, x_idx] |= discovered

        reward = reward * alive
        self.r[:, self.g, self.t, :] = reward
        self.perf[:, self.g, :] += reward

        self.t += 1

    def select_by_performance(self, beta: float) -> np.ndarray:
        """Selects agents based on performance using payoff-biased copying.

        Args:
            beta: Selection strength parameter.

        Returns:
            Selected agent indices, shape (R, N).
        """
        assert self.perf is not None
        probs = softmax(beta * self.perf[:, self.g, :], axis=1)         # [R,N]
        # repeat for each agent
        probs = probs[:, None, :].repeat(self.N, axis=1)                  # [R,N,N]
        selected_idx = sample_categorical(self.rng, probs).astype(np.int32)                 # [R,N]
        return selected_idx


    def teach(self) -> np.ndarray:
        """Transmits strategies from the teacher and depth from parent (for cultural evolution).
        
        Returns:
            Strategies and depths, shape (R, N, X) and (R, N).
        """
        teacher_idx = self.select_by_performance(self.agent_config.beta_teacher) # [R,N]
        r_idx = np.arange(self.R)[:, None]
        teacher_K = self.K[r_idx, self.g, teacher_idx, :]
        
        # For depth inheritance, use separate parent selection if learn_d is True
        if self.agent_config.learn_d:
            parent_idx = self.select_by_performance(self.agent_config.beta_evo) # [R,N]
            teacher_d = self.d_i[r_idx, self.g, parent_idx]
        else:
            teacher_d = self.d_i[r_idx, self.g, teacher_idx]
        
        return teacher_K, teacher_d


    def learn(self, teacher_K: np.ndarray, teacher_d: np.ndarray) -> None:
        """Learns strategies from the teacher.

        Args:
            teacher_K: Strategies from the teacher, shape (R, N, X).
        """
        self.K[:, self.g, :, :] = teacher_K
        self.K[:, self.g, :, 0] = True
        if self.agent_config.learn_d:
            self.d_i[:, self.g, :] = teacher_d


    def next_generation(self):
        """Advances to the next generation.
        """
        self.g += 1
        self.t = 0
        print(f"Generation {self.g} started")


    def save(self, path: str) -> None:
        """Saves the agent population to a file.

        Args:
            path: Path to save the agent population.
        """
        K_df = using_multiindex(self.K, columns=["R", "G", "N", "X"], value_name="K")
        perf_df = using_multiindex(self.perf, columns=["R", "G", "N"], value_name="perf")
        x_df = using_multiindex(self.x, columns=["R", "G", "T", "N"], value_name="x")
        m_df = using_multiindex(self.m, columns=["R", "G", "T", "N"], value_name="m")
        r_df = using_multiindex(self.r, columns=["R", "G", "T", "N"], value_name="r")
        
        os.makedirs(path, exist_ok=True)
        K_df.to_parquet(os.path.join(path, "K.parquet"))
        perf_df.to_parquet(os.path.join(path, "perf.parquet"))
        x_df.to_parquet(os.path.join(path, "x.parquet"))
        m_df.to_parquet(os.path.join(path, "m.parquet"))
        r_df.to_parquet(os.path.join(path, "r.parquet"))