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
    - K: repertoire values (max observed payoff per strategy)                          # [R,G,N,S]
    - b: best-known strategy index per task                                            # [R,G,N,P]
    - best_r: best observed payoff per task                                            # [R,G,N,P]
    - perf: performance accumulator (sum of rewards)                                   # [R,G,N]
    """

    R: int # number of replications
    G: int # number of generations
    N: int | None = None # number of agents
    P: int # number of tasks
    S: int # number of strategies
    L: int # maximum strategy length

    g: int = 0 # current generation index

    agent_config: AgentConfig
    rng: np.random.Generator

    T_i: np.ndarray | None = None # [R,G,N] lifetime trials per agent
    d_i: np.ndarray | None = None # [R,G,N] exploration depth (max length for novel prefixes)
    eps_i: np.ndarray | None = None # [R,G,N] exploration probability
    AT: np.ndarray | None = None # [R,G,N] agent type indices

    K: np.ndarray | None = None # [R,G,N,S] repertoire (known strategies and their payoffs)
    b: np.ndarray | None = None # [R,G,N,P] best-known strategy per task
    best_r: np.ndarray | None = None # [R,G,N,P] best-known payoff per task
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

        self.K = np.full((self.R, self.G, self.N, self.S), -np.inf, dtype=float)
        self.K[..., 0] = 0 # safe strategy payoff
        self.b = np.zeros((self.R, self.G, self.N, self.P), dtype=np.int32)  # default to safe strategy
        self.best_r = np.full((self.R, self.G, self.N, self.P), -np.inf, dtype=float)
        self.perf = np.zeros((self.R, self.G, self.N), dtype=float)
        self.pow2 = np.power(2, np.arange(self.L + 1))


    def is_alive(self, t: int, g: int) -> np.ndarray:
        """
        Returns alive mask for this trial.
        alive: (R,N) bool                                                           # [R,N]
        """
        assert self.T_i is not None
        return (t < self.T_i[:, g, :])                                   # [R,N]

    def do_explore(self) -> np.ndarray:
        """
        Samples exploration mode indicators.
        do_explore: (R,N) bool                                                      # [R,N]
        """
        assert self.eps_i is not None
        do_explore = (self.rng.random(size=(self.R, self.N)) < self.eps_i[:, self.g, :])       # [R,N]
        return do_explore

    def explore(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Samples a candidate strategy index for exploration.

        Note: in this vectorized prototype, exploration depth controls the *breadth* of the
        strategy index range searched: max_idx = min(2^d_i, S).

        Returns:
            s_idx: (R,N) int32                                                      # [R,N]
            is_known: (R,N) bool                                                    # [R,N]
        """
        assert self.d_i is not None and self.K is not None and self.pow2 is not None

        max_idx = np.minimum(self.pow2[self.d_i[:, self.g, :]], self.S).astype(np.int32)  # [R,N]
        max_idx = np.maximum(max_idx, 1)  # ensure valid upper bound
        s_idx = self.rng.integers(0, max_idx, size=(self.R, self.N), dtype=np.int32)      # [R,N]

        r_idx = np.arange(self.R)[:, None]
        n_idx = np.arange(self.N)[None, :]
        is_known = self.K[r_idx, self.g, n_idx, s_idx] > -np.inf                           # [R,N]
        return s_idx, is_known

    def exploit(self, p_idx: np.ndarray) -> np.ndarray:
        """Returns current best strategy index for each agent on sampled task."""
        assert self.b is not None
        r_idx = np.arange(self.R)[:, None]
        n_idx = np.arange(self.N)[None, :]
        return self.b[r_idx, self.g, n_idx, p_idx].astype(np.int32)                        # [R,N]

    def act(self, p_idx: np.ndarray) -> np.ndarray:
        """
        Returns strategy indices for each agent.
        s_idx: (R,N) int32                                                          # [R,N]
        """
        try_explore = self.do_explore()
        explore_idx, is_known = self.explore()
        do_exploit = is_known | ~try_explore
        exploit_idx = self.exploit(p_idx)
        s_idx = np.where(do_exploit, exploit_idx, explore_idx).astype(np.int32)
        return s_idx

    def update(self, p_idx: np.ndarray, s_idx: np.ndarray, reward: np.ndarray) -> None:
        """
        Updates within-lifetime state for generation g given observed rewards.

        - Updates repertoire values K for the chosen strategy.
        - Updates per-task best strategy b / best_r.
        - Accumulates performance perf.
        """
        assert self.K is not None and self.perf is not None and self.b is not None and self.best_r is not None

        r_idx = np.arange(self.R)[:, None]
        n_idx = np.arange(self.N)[None, :]

        # repertoire update: keep max observed payoff for (agent,strategy)
        old_k = self.K[r_idx, self.g, n_idx, s_idx]
        self.K[r_idx, self.g, n_idx, s_idx] = np.maximum(old_k, reward)

        # performance accumulator
        self.perf[:, self.g, :] += reward

        # best-by-task update
        old_best = self.best_r[r_idx, self.g, n_idx, p_idx]
        better = reward > old_best
        self.best_r[r_idx, self.g, n_idx, p_idx] = np.where(better, reward, old_best)
        old_b = self.b[r_idx, self.g, n_idx, p_idx]
        self.b[r_idx, self.g, n_idx, p_idx] = np.where(better, s_idx, old_b).astype(np.int32)


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

    def export_strategies(self, teacher_idx: np.ndarray) -> np.ndarray:
        """Export strategies from teachers to learners.

        Args:
            teacher_idx: Teacher indices, shape (R, N).

        Returns:
            Strategies, shape (R, N, S).
        """
        r_idx = np.arange(self.R)[:, None]
        teacher_K = self.K[r_idx, self.g, teacher_idx, :]               # [R,N,S]
        return teacher_K

    def teach(self) -> np.ndarray:
        """Transmits strategies from the teacher.
        
        Returns:
            Strategies, shape (R, N, S).
        """
        teacher_idx = self.select_teacher() # [R,N]
        return self.export_strategies(teacher_idx) # [R,N,S]


    def learn(self, teacher_K: np.ndarray) -> None:
        """Learns strategies from the teacher.

        Args:
            teacher_K: Strategies from the teacher, shape (R, N, S).
        """
        self.K[:, self.g, :, :] = teacher_K
        self.K[:, self.g, :, 0] = 0


    def next_generation(self):
        """Advances to the next generation.
        """
        self.g += 1


    def save(self, path: str) -> None:
        """Saves the agent population to a file.

        Args:
            path: Path to save the agent population.
        """
        K_df = array_to_df(self.K, index=["R", "G", "N", "S"], columns=["K"])
        b_df = array_to_df(self.b, index=["R", "G", "N", "P"], columns=["b"])
        best_r_df = array_to_df(self.best_r, index=["R", "G", "N", "P"], columns=["best_r"])
        perf_df = array_to_df(self.perf, index=["R", "G", "N"], columns=["perf"])
        
        os.makedirs(path, exist_ok=True)
        K_df.to_parquet(os.path.join(path, "K.parquet"))
        b_df.to_parquet(os.path.join(path, "b.parquet"))
        best_r_df.to_parquet(os.path.join(path, "best_r.parquet"))
        perf_df.to_parquet(os.path.join(path, "perf.parquet"))