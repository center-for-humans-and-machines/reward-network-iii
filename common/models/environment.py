from typing import Optional, List, Union
from pydantic import BaseModel, validator, Extra


class Level(BaseModel):
    idx: int
    color: str
    is_start: Optional[bool] = None
    n_nodes: Optional[int] = None
    min_n_nodes: int = 1
    max_n_nodes: Optional[int] = None

    class Config:
        extra = Extra.forbid


class Rewards(BaseModel):
    idx: int
    reward: int
    color: str

    class Config:
        extra = Extra.forbid


class Edge(BaseModel):
    from_level: int
    to_levels: List[int]
    rewards: List[int]

    class Config:
        extra = Extra.forbid


class Environment(BaseModel):
    n_networks: int
    seed: int
    levels: List[Level]
    rewards: List[Rewards]
    edges: List[Edge]
    n_edges_per_node: float
    n_nodes: int
    n_steps: int
    n_losses: int

    @validator("levels", pre=True)
    def val_levels(cls, levels):
        return [{**l, "idx": i} for i, l in enumerate(levels)]

    @validator("rewards", pre=True)
    def val_rewards(cls, rewards):
        return [{**r, "idx": i} for i, r in enumerate(rewards)]

    class Config:
        extra = Extra.forbid
