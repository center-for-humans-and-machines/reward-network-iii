from __future__ import annotations
import argparse
import numpy as np
from abm.task import TaskEnv, TaskConfig
from abm.agents import AgentPop, AgentConfig
from abm.utils import load_config


def run_simulation(R: int, G: int, N: int, P: int, L: int, S: int, seed: int, agent_config: dict, task_config: dict):
    rng = np.random.default_rng(seed)
    agent_config_obj = AgentConfig(**agent_config)
    task_config_obj = TaskConfig(**task_config)
    agent_pop = AgentPop(R=R, G=G, N=N, P=P, S=S, L=L, rng=rng, agent_config=agent_config_obj)
    task_env = TaskEnv(R=R, P=P, S=S, L=L, N=N, rng=rng, task_config=task_config_obj)
    for g in range(G):
        if g > 0:
            agent_pop.learn(student_K, student_d)
        alive = agent_pop.is_alive()
        while alive.any():
            x_idx = agent_pop.act()
            reward = task_env.step(alive, x_idx)
            agent_pop.update(alive, x_idx, reward)
            alive = agent_pop.is_alive()

        teacher_K, teacher_d = agent_pop.teach()
        student_K, student_d = task_env.transmit(teacher_K, teacher_d)

        agent_pop.next_generation()
    
    return agent_pop
    

if __name__ == "__main__":
    # get config and output path from command line
    parser = argparse.ArgumentParser(description="Run simulation")
    parser.add_argument("-c", "--config", help="Path to the config YAML file", required=True)
    parser.add_argument("-o", "--output", help="Path to the output directory", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    agent_pop = run_simulation(**config)
    agent_pop.save(args.output)