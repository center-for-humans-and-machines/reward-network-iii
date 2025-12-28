from __future__ import annotations
import argparse
import numpy as np
from task import TaskEnv, TaskConfig
from agents import AgentPop, AgentConfig
from utils import load_config
from pydantic import BaseModel, Field, field_validator, model_validator


class SimulationConfig(BaseModel):
    """Pydantic model for validating simulation configuration."""
    R: int = Field(gt=0, description="Number of replications")
    G: int = Field(gt=0, description="Number of generations")
    N: int = Field(gt=0, description="Number of agents")
    P: int = Field(gt=0, description="Number of tasks")
    L: int = Field(gt=0, description="Maximum strategy length")
    S: int | None = Field(default=None, gt=0, description="Number of latent strategies (optional, will be computed from strategies if not provided)")
    seed: int = Field(description="Random seed")
    agent_config: AgentConfig
    task_config: TaskConfig



def run_simulation(c: SimulationConfig):
    rng = np.random.default_rng(c.seed)
    agent_pop = AgentPop(R=c.R, G=c.G, N=c.N, P=c.P, L=c.L, rng=rng, agent_config=c.agent_config)
    task_env = TaskEnv(R=c.R, P=c.P, L=c.L, N=c.N, rng=rng, task_config=c.task_config)
    for g in range(c.G):
        if g > 0:
            agent_pop.learn(student_K, student_d)
        alive = agent_pop.is_alive()
        while alive.any():
            x_idx = agent_pop.act()
            reward = task_env.step(alive, x_idx)
            agent_pop.update(alive, x_idx, reward)
            alive = agent_pop.is_alive()

        if g < c.G - 1:  # Only teach/transmit if not the last generation
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
    config = SimulationConfig(**config)
    agent_pop = run_simulation(config)
    agent_pop.save(args.output)