from __future__ import annotations
import numpy as np
from utils import compute_rle, sigmoid
from dataclasses import dataclass
from typing import Any, Literal, Optional
from pydantic import BaseModel


class TaskConfig(BaseModel):
    r_safe: float | None = None
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
    """

    # dimensions
    R: int # number of replications
    P: int # number of tasks
    L: int # maximum strategy length
    N: int # number of agents

    rng: np.random.Generator

    task_config: TaskConfig

    X: int | None = None # number of all possible strategies
    S: int | None = None # number of latent strategies

    # latent strategies
    x_latent: np.ndarray | None = None    # [S] is latent strategy
    x_len: np.ndarray | None = None    # [S] length of latent strategies
    x_cost: np.ndarray | None = None    # [S] cost of latent strategies
    x_bonus: np.ndarray | None = None    # [S] bonus of latent strategies
    x_q: np.ndarray | None = None      # [S] learnability of latent strategies

    # applicability and payoff lookup
    W: np.ndarray | None = None        # [R,P,S] applicability matrix for latent strategies on tasks

    
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

        if '0' not in strategies:
            strategies.append('0')

        self.X = 2 ** self.L
        self.S = len(strategies)
        assert self.L >= max(len(strategy) for strategy in strategies)
        self.s_idx = np.array([int(strategy, 2) for strategy in strategies], dtype=np.int8)
        self.x_len = np.zeros(self.X, dtype=np.int32)
        self.x_latent = np.full(self.X, -1, dtype=bool)
        self.x_q = np.zeros(self.X, dtype=float)
        
        for idx in range(self.X):
            self.x_latent[idx] = idx in self.s_idx
            self.x_len[idx] = 0 if idx == 0 else idx.bit_length()
            rle = compute_rle(idx)
            self.x_q[idx] = sigmoid(self.task_config.alpha - self.task_config.gamma * rle)
        self.x_cost = self.task_config.c_cost * self.x_len
        self.x_bonus = self.task_config.r_scale * (self.task_config.lam ** self.x_len) * self.x_latent

        print(self.x_cost)
        print(self.x_bonus)
        print(self.x_latent)
        print(self.s_idx)

        # self.build_applicability()

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

    # def build_applicability(self) -> np.ndarray:
    #     """
    #     Builds applicability W[r,p,s] ∈ {0,1}. Safe strategy is always applicable.
    #     """
    #     self.W = np.zeros((self.R, self.P, self.X), dtype=bool)                                        # [R,P,S]
    #     self.W[..., 0] = True # safe strategy is always applicable

    #     n_applicable = int(self.task_config.p_applicable * self.S)

    #     if self.task_config.mode == "sparse":
    #         for r in range(self.R):
    #             for p in range(self.P):
    #                 s_sample = self.rng.choice(self.s_idx[1:], size=n_applicable, replace=False) # exclude safe strategy
    #                 self.W[r, p, s_sample] = True


    def step(self, alive: np.ndarray, x_idx: np.ndarray) -> np.ndarray:
        """
        Samples a task and computes the reward for applying strategies x_idx on the task.
        Returns:
            p_idx: task indices                                                         # [R,N]
            reward: reward for applying strategies x_idx on the task.                    # [R,N]
        """
        # p_idx = self.rng.integers(0, self.P, size=(self.R, self.N), dtype=np.int32)
        
        R = self.R
        cost = self.x_cost[x_idx] # [R,N]
        bonus = self.x_bonus[x_idx] # [R,N]
        discovered = (self.rng.random(size=(self.R, self.N)) < self.task_config.p_applicable)
        reward = np.where(discovered, bonus - cost, -cost)
        if self.task_config.r_safe is not None:
            reward = np.where(x_idx == 0, self.task_config.r_safe, reward)
        reward = reward * alive
        return reward                        # [R,N]


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
        p_learn = self.x_q[None, None, :] # [R,N,X]
        is_success = self.rng.random(size=(self.R, self.N, self.X)) < p_learn # [R,N,X]
        student_K = np.where(is_success, teacher_K, False) # [R,N,X]
        student_K[..., 0] = True # safe strategy is always learned

        min_d = 0
        max_d = self.L
        d_mutation = self.rng.integers(-1, 2, size=(self.R, self.N), dtype=np.int32)
        student_d = np.clip(teacher_d + d_mutation, min_d, max_d)

        return student_K, student_d # [R,N,X], [R,N]
