from __future__ import annotations
import numpy as np
from abm.task import TaskEnv
from abm.agents import AgentPop
from abm.utils import load_config


def run_simulation(R: int, G: int, N: int, P: int, J: int, L: int, S: int, seed: int, agent_config: dict, task_config: dict):
    rng = np.random.default_rng(seed)
    agent_pop = AgentPop(R=R, G=G, N=N, P=P, J=J, L=L, seed=seed, **agent_config)
    task_env = TaskEnv(R=R, P=P, S=S, L=L, seed=seed, **task_config)
    for g in range(G):
        p_idx = task_env.sample_tasks(rng, N)
        s_idx = agent_pop.act(g, p_idx, rng)
        reward = task_env.step(p_idx, s_idx)
        agent_pop.update(g, p_idx, s_idx, reward)
        agent_pop.transmit(g, rng)



if __name__ == "__main__":
    config = load_config("config.yml")
    run_simulation(**config)