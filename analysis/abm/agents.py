from __future__ import annotations

import numpy as np
from abm.utils import sample_categorical, softmax
from utils.array_to_df import array_to_df
from dataclasses import dataclass
from pydantic import BaseModel
import os

class AgentConfig(BaseModel):
    beta_teacher: float
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
    - K: repertoire values (max observed payoff per strategy)                          # [R,G,N,X]
    - b: best-known strategy index per task                                            # [R,G,N,P]
    - best_r: best observed payoff per task                                            # [R,G,N,P]
    - perf: performance accumulator (sum of rewards)                                   # [R,G,N]
    """

    R: int # number of replications
    G: int # number of generations
    N: int | None = None # number of agents
    P: int # number of tasks
    L: int # maximum strategy length
    X: int | None = None # number of all possible strategies

    g: int = 0 # current generation index
    t: int = 0 # current trial index

    agent_config: AgentConfig
    rng: np.random.Generator

    T_i: np.ndarray | None = None # [R,G,N] lifetime trials per agent
    d_i: np.ndarray | None = None # [R,G,N] exploration depth (max length for novel prefixes)
    eps_i: np.ndarray | None = None # [R,G,N] exploration probability
    phi_i: np.ndarray | None = None # [R,G,N] exploration probability
    AT: np.ndarray | None = None # [R,G,N] agent type indices

    K: np.ndarray | None = None # [R,G,N,X] repertoire (known strategies and their payoffs)
    perf: np.ndarray | None = None # [R,G,N] performance accumulator

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
        self.X = 2 ** self.L
        self.N = int(sum(self.agent_config.N_0))
        assert self.N > 0
        # usually we keep population size constant across generations
        assert int(sum(self.agent_config.N_o)) == self.N
        
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
        self.d_i = np.array(self.agent_config.d)[AT]
        self.eps_i = np.array(self.agent_config.eps)[AT]
        self.phi_i = np.array(self.agent_config.phi)[AT]
        
        self.K = np.zeros((self.R, self.G, self.N, self.X), dtype=bool)
        self.perf = np.zeros((self.R, self.G, self.N), dtype=float)
        self.pow2 = np.power(2, np.arange(self.L + 1))


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
        """Returns a random known strategy index."""
        x_idx = self.rng.choice(self.X, p=self.K[:,self.g], size=(self.R, self.N), axis=2, replace=False)
        return x_idx

    def act(self) -> np.ndarray:
        """
        Returns strategy indices for each agent.
        s_idx: (R,N) int32                                                          # [R,N]
        """
        go_save = (self.rng.random(size=(self.R, self.N)) < self.eps_i[:, self.g, :])
        go_exploit = (self.rng.random(size=(self.R, self.N)) < self.phi_i[:, self.g, :])

        x_idx = np.zeros((self.R, self.N), dtype=np.int32)
        explore_idx = self.explore()
        exploit_idx = self.exploit()

        x_idx = np.where(~go_save & go_exploit, exploit_idx, x_idx)
        x_idx = np.where(~go_save & ~go_exploit, explore_idx, x_idx)
        return x_idx

    def update(self, alive: np.ndarray, x_idx: np.ndarray, reward: np.ndarray) -> None:
        """
        Updates within-lifetime state for generation g given observed rewards.

        - Updates repertoire values K for the chosen strategy.
        - Updates per-task best strategy b / best_r.
        - Accumulates performance perf.
        """
        assert self.K is not None and self.perf is not None and self.b is not None and self.best_r is not None

        r_idx = np.arange(self.R)[:, None]
        n_idx = np.arange(self.N)[None, :]

        discovered = reward > 0 & alive
        self.K[r_idx, self.g, n_idx, x_idx] &= discovered

        # performance accumulator
        self.perf[:, self.g, :] += reward * alive

        self.t += 1

    def select_teacher(self) -> np.ndarray:
        """Selects a teacher for each agent.

        Returns:
            Teacher indices, shape (R, N).
        """
        assert self.perf is not None
        teach_probs = softmax(self.agent_config.beta_teacher * self.perf[:, self.g, :], axis=1)         # [R,N]
        # repeat for each agent
        teach_probs = teach_probs[:, None, :].repeat(self.N, axis=1)                  # [R,N,N]
        teacher_idx = sample_categorical(self.rng, teach_probs).astype(np.int32)                 # [R,N]
        return teacher_idx


    def teach(self) -> np.ndarray:
        """Transmits strategies from the teacher.
        
        Returns:
            Strategies, shape (R, N, X).
        """
        teacher_idx = self.select_teacher() # [R,N]
        r_idx = np.arange(self.R)[:, None]
        teacher_K = self.K[r_idx, self.g, teacher_idx, :]
        teacher_d = self.d_i[r_idx, self.g, teacher_idx]
        return teacher_K, teacher_d


    def learn(self, teacher_K: np.ndarray, teacher_d: np.ndarray) -> None:
        """Learns strategies from the teacher.

        Args:
            teacher_K: Strategies from the teacher, shape (R, N, X).
        """
        self.K[:, self.g, :, :] = teacher_K
        self.K[:, self.g, :, 0] = True
        if self.learn_d:
            self.d_i[:, self.g, :] = teacher_d


    def next_generation(self):
        """Advances to the next generation.
        """
        self.g += 1


    def save(self, path: str) -> None:
        """Saves the agent population to a file.

        Args:
            path: Path to save the agent population.
        """
        K_df = array_to_df(self.K, index=["R", "G", "N", "X"], columns=["K"])
        perf_df = array_to_df(self.perf, index=["R", "G", "N"], columns=["perf"])
        
        os.makedirs(path, exist_ok=True)
        K_df.to_parquet(os.path.join(path, "K.parquet"))
        perf_df.to_parquet(os.path.join(path, "perf.parquet"))