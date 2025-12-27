from __future__ import annotations
import numpy as np
from abm.utils import compute_rle, sigmoid
from dataclasses import dataclass
from typing import Any, Literal, Optional
from pydantic import BaseModel

class TaskConfig(BaseModel):
    r_safe: float
    c_cost: float
    r_scale: float
    lam: float

    strategies: Optional[list[str]]
    strategy_distribution: Literal["uniform", "fixed"]
    p_applicable: float
    mode: Literal["sparse"]
    alpha: float
    gamma: float

@dataclass
class TaskEnv:
    """
    Task family and payoff logic (vectorized over replications and tasks).

    Stored arrays:
    - Latent strategies: s_bits, s_len, s_rle                                    # [S,L], [S], [S]
    - Applicability: W[r,p,s]                                                     # [R,P,S]
    - Strategy payoff lookup: Rps[r,p,s] computed once per run                     # [R,P,S]
    """

    # dimensions
    R: int # number of replications
    P: int # number of tasks
    S: int # number of latent strategies
    L: int # maximum strategy length
    X: int | None = None # number of all possible strategies
    N: int # number of agents

    rng: np.random.Generator

    task_config: TaskConfig

    # latent strategies
    s_idx: np.ndarray | None = None    # [S] index of latent strategies
    s_len: np.ndarray | None = None    # [S] length of latent strategies
    s_rle: np.ndarray | None = None    # [S] run-length encoding of latent strategies
    s_q: np.ndarray | None = None      # [S] learnability of latent strategies

    # applicability and payoff lookup
    W: np.ndarray | None = None        # [R,P,S] applicability matrix for latent strategies on tasks
    Rps: np.ndarray | None = None      # [R,P,S] payoff matrix for latent strategies on tasks
    x_to_s: np.ndarray | None = None  # [S] mapping from strategies to latent strategies


    
    def __post_init__(self):
        """
        Creates a TaskEnv from a list of strategies.
        """
        if self.task_config.strategy_distribution == "fixed":
            assert self.task_config.strategies is not None
            strategies = self.task_config.strategies
        elif self.task_config.strategy_distribution == "uniform":
            strategies = self.strategies_from_distribution()
        else:
            raise ValueError(f"Unknown strategy distribution: {self.task_config.strategy_distribution}")

        if '' not in strategies:
            strategies.append('')

        self.X = 2 ** self.L
        assert self.S >= len(strategies)
        assert self.L >= max(len(strategy) for strategy in strategies)
    
        self.s_idx = np.full(self.S, -1, dtype=np.int8)
        self.s_len = np.zeros(self.S, dtype=np.int32)
        self.s_rle = np.zeros(self.S, dtype=np.int32)
        self.s_q = np.zeros(self.S, dtype=float)

        self.x_to_s = np.full(self.X, -1, dtype=np.int8)
        
        for i, strategy in enumerate(strategies):
            idx = int(strategy, 2)
            self.s_idx[i] = idx
            # self.s_len[i] = idx.bit_length()
            rle = compute_rle(idx)
            self.s_q[i] = sigmoid(self.task_config.alpha - self.task_config.gamma * rle)
            self.x_to_s[idx] = i
        self.build_applicability()
        self.compute_strategy_payoffs()

    def strategies_from_distribution(self) -> list[str]:
        """
        Creates a list of strategies from a distribution.
        Returns list of binary strings (e.g., "101", "1101").
        """
        if self.task_config.strategy_distribution == "uniform":
            strategies = []
            for l in range(1, self.L + 1):
                bits = self.rng.integers(0, 2, size=l, dtype=np.int8)
                strategy_str = ''.join(str(bit) for bit in bits)
                strategies.append(strategy_str)
            return strategies
        else:
            raise ValueError(f"Unknown distribution: {self.task_config.strategy_distribution}")

    def build_applicability(self) -> np.ndarray:
        """
        Builds applicability W[r,p,s] ∈ {0,1}. Safe strategy is always applicable.
        """
        assert self.s_len is not None, "Strategies must be built first."

        self.W = np.zeros((self.R, self.P, self.S), dtype=bool)                                        # [R,P,S]
        self.W[..., 0] = True # safe strategy is always applicable

        if self.task_config.mode == "sparse":
            X = np.argsort(self.rng.random(size=(self.R, self.P, self.S - 1)), axis=-1)  # [R,P,S-1]
            self.W[..., 1:] = X < self.task_config.p_applicable * (self.S - 1) # [R,P,S-1]

    def compute_strategy_bonus(self) -> np.ndarray:
        """
        Precomputes payoff lookup Rps[r,p,s] for applying a *strategy* s on task p.  # [R,P,S]
        Non-strategy prefixes are handled via nonstrategy_payoff(length).
        """
        assert self.W is not None and self.s_len is not None

                                          # [1,1,S]
        bonus = np.einsum("rps,s->rps", self.W.astype(float), reward_vec)          # [R,P,S]
        self.Rps = base + bonus                                                         # [R,P,S]
        self.Rps[..., 0] = self.task_config.r_safe


    def step(self, alive: np.ndarray, x_idx: np.ndarray) -> np.ndarray:
        """
        Samples a task and computes the reward for applying strategies x_idx on the task.
        Returns:
            p_idx: task indices                                                         # [R,N]
            reward: reward for applying strategies x_idx on the task.                    # [R,N]
        """
        assert self.Rps is not None

        p_idx = self.sample_tasks()
        
        R = self.R
        x_len = np.log2(x_idx + 1).astype(np.int32)
        cost = self.task_config.c_cost * x_len # [R,N]
        s_idx = self.x_to_s[x_idx]
        discovered = s_idx != -1
        s_idx = np.where(discovered, s_idx, 0) # Hack to handle non-strategies

        bonus = self.Rps[np.arange(R)[:, None], p_idx, s_idx]
        bonus = np.where(discovered, bonus, 0)

        reward = - cost + bonus

        reward = reward * alive

        return reward                        # [R,N]


    def sample_tasks(self) -> np.ndarray:
        """
        Samples tasks uniformly.
        p_idx: (R,N) int32                                                          # [R,N]
        """
        return self.rng.integers(0, self.P, size=(self.R, self.N), dtype=np.int32)       # [R,N]


    def transmit(self, teacher_K: np.ndarray, teacher_d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Transmits strategies from the teacher to the learner.
        Args:
            teacher_K: Strategies from the teacher, shape (R, N, X).
            teacher_d: Exploration depths from the teacher, shape (R, N).
        Returns:
            student_K: Strategies learned by the learner, shape (R, N, X).
            student_d: Exploration depths learned by the learner, shape (R, N).
        """
        # create p of shape (R, N, X)
        p_learn = self.s_q[None, None, :] # [R,N,X]
        is_success = self.rng.random(size=(self.R, self.N, self.X)) < p_learn # [R,N,X]
        student_K = np.where(is_success, teacher_K, False) # [R,N,X]
        student_K[..., 0] = True # safe strategy is always learned

        min_d = 0
        max_d = self.L
        d_mutation = self.rng.integers(-1, 2, size=(self.R, self.N), dtype=np.int32)
        student_d = np.clip(teacher_d + d_mutation, min_d, max_d)

        return student_K, student_d # [R,N,X], [R,N]
