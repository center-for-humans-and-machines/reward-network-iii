from __future__ import annotations

import numpy as np
from abm.utils import sample_categorical, softmax
from dataclasses import dataclass
from pydantic import BaseModel


class AgentConfig(BaseModel):
    beta_teacher: float
    K_transmit: int
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

    Stored arrays:
    - T_i: lifetime trials per agent                                                 # [R,N]
    - d_i: exploration depth (max length for novel prefixes)                          # [R,N]
    - K: repertoire (known strategies)                                                # [R,N,S]
    - b: best-known strategy index per task                                            # [R,N,P]
    - perf_g: performance accumulator for current generation                           # [R,N]
    """

    R: int # number of replications
    G: int # number of generations
    N: int | None = None # number of agents
    P: int # number of tasks
    S: int # number of strategies
    L: int # maximum strategy length

    agent_config: AgentConfig

    T_i: np.ndarray | None = None # [R,G,N] lifetime trials per agent
    d_i: np.ndarray | None = None # [R,G,N] exploration depth (max length for novel prefixes)
    eps_i: np.ndarray | None = None # [R,G,N] exploration probability
    AT: np.ndarray | None = None # [R,G,N] agent type indices

    K: np.ndarray | None = None # [R,G,N,S] repertoire (known strategies and their payoffs)
    perf: np.ndarray | None = None # [R,G,N] performance accumulator

    #utils
    pow2: np.ndarray | None = None # [L+1] 2^l for l<=L

    def __post_init__(self) -> None:
        assert len(self.agent_config.agent_types) == len(self.agent_config.T) == len(self.agent_config.d) == len(self.agent_config.N_0) == len(self.agent_config.N_o)
        assert sum(self.agent_config.N_o) == len(self.agent_config.N_0)

        self.N = sum(self.agent_config.N_0)
        
        # initialize agent type indices for each generation
        AT = np.zeros((self.R, self.G, self.N), dtype=np.int32)
        at_0 = np.array([[i]*n for i, n in enumerate(self.agent_config.N_0)])
        at_o = np.array([[i]*n for i, n in enumerate(self.agent_config.N_o)])
        AT[:,0,:] = at_0[None,:]
        AT[:,1:,:] = at_o[None,None,:]
        self.AT = AT

        self.T_i = np.array(self.agent_config.T)[AT]
        self.d_i = np.array(self.agent_config.d)[AT]
        self.eps_i = np.array(self.agent_config.eps)[AT]

        self.K = np.full((self.R, self.G, self.N, self.S), -np.inf, dtype=float)
        self.K[..., 0] = 0 # safe strategy payoff
        self.perf_g = np.zeros((self.R, self.G, self.N), dtype=float)
        self.pow2 = np.power(2, np.arange(self.L + 1))


    def is_alive(self, t: int, g: int) -> np.ndarray:
        """
        Returns alive mask for this trial.
        alive: (R,N) bool                                                           # [R,N]
        """
        return (t < self.T_i[g])                                   # [R,N]

    def do_explore(
        self,
        g: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Samples exploration mode indicators.
        do_explore: (R,N) bool                                                      # [R,N]
        """
        do_explore = (rng.random(size=(self.R, self.N)) < self.eps_i[g])       # [R,N]
        return do_explore

    def explore(self, g: int, p_idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Creates a prefix defined by length and index                                                   # [R,N]
        length: (R,N) int32                                                          # [R,N]
        idx: (R,N) int32                                                          # [R,N]
        """
        length = self.pow2[self.d_i[g]]
        idx = rng.integers(0, length, size=(self.R, self.N))
        is_known = self.K[g, :, p_idx, idx] > -np.inf
        return idx, is_known

    def exploit(self, g: int, p_idx: np.ndarray) -> np.ndarray:
        """Returns current best strategy index for each agent on sampled task."""
        return np.argmax(self.K[g, :, p_idx], axis=-1)

    def act(self, g: int, p_idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Returns strategy indices for each agent.
        s_idx: (R,N) int32                                                          # [R,N]
        """
        do_explore = self.do_explore(g, rng)
        explore_idx, is_known = self.explore(g, p_idx, rng)
        do_exploit = is_known | ~do_explore
        exploit_idx = self.exploit(g, p_idx)
        s_idx = np.where(do_exploit, exploit_idx, explore_idx)
        return s_idx

    def update(self, g: int, p_idx: np.ndarray, s_idx: np.ndarray, reward: np.ndarray) -> None:
        """
        Updates the repertoire with the new prefix if it is a match.
        """
        # update repertoire with new prefix if reward is positive
        self.K[g, :, p_idx, s_idx] = np.where(reward > 0, reward, self.K[g, :, p_idx, s_idx])
        self.perf[g, :, :] += reward


    def select_teacher(self, g: int, rng: np.random.Generator) -> np.ndarray:
        """
        Selects a teacher for each agent.
        teacher_idx: (R,N) int32                                                      # [R,N]
        """
        teach_probs = softmax(self.agent_config.beta_teacher * self.perf[g, :, :], axis=1)         # [R,N]
        # repeat for each agent
        teach_probs = teach_probs[:, None, :].repeat(self.N, axis=1)                  # [R,N,N]
        teacher_idx = sample_categorical(rng, teach_probs)                            # [R,N]
        return teacher_idx


    def export_strategies(self, g: int, teacher_idx: np.ndarray) -> np.ndarray:
        """
        Exports strategies from the teacher to the learner.
        strategies: (R,N(learners),S) bool                                                     # [R,N,S]
        """
        # TODO: generation is not yet implemented
        teacher_K = self.K[np.arange(self.R)[:, None], teacher_idx, :]               # [R,N,S]

        # select top-K strategies
        teacher_K_rank = np.argsort(-teacher_K, axis=2)                                  # [R,N,S]
        teacher_K = np.where(teacher_K_rank < self.agent_config.K_transmit, teacher_K, -np.inf)            # [R,N,S]
        return teacher_K


    def transmit(self, g: int, rng: np.random.Generator):
        """
        Transmits strategies from the teacher to the learner.
        strategies: (R,N(learners),S) bool                                                     # [R,N,S]
        """
        teacher_idx = self.select_teacher(g, rng) # [R,N]
        teacher_K = self.export_strategies(g, teacher_idx) # [R,N,S]
        # TODO: generation is not yet implemented
        self.K[g+1, :, :, :] = teacher_K
