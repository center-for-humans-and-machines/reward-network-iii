import numpy as np
import logging
from pydantic import BaseModel


class RewardNetworkParams(BaseModel):
    n_steps: int


class Reward_Network:
    def __init__(self, network, params: RewardNetworkParams):
        """_summary_
        Args:
            network (dict): a single network object
            params (dict): parameters for solving the networks eg n_steps, possible rewards
        """

        params = dict(RewardNetworkParams(**params))

        # -------------
        # assert tests
        # -------------

        # reward network information from json file
        self.network = network

        # initial reward and step values
        self.INIT_REWARD = 0
        self.INIT_STEP = 0

        self.MAX_STEP = params["n_steps"]  # 8

        # network info
        self.id = self.network["network_id"]
        self.nodes = self.network["nodes"]
        self.action_space = self.network["edges"]
        self.possible_rewards = list(set([e["reward"] for e in self.network["edges"]]))
        self.reward_range = (
            min(self.possible_rewards) * self.MAX_STEP,
            max(self.possible_rewards) * self.MAX_STEP,
        )

    def reset(self):
        # Reset the state of the environment to an initial state
        self.reward_balance = self.INIT_REWARD
        self.step_counter = self.INIT_STEP
        self.max_level = 0
        self.is_done = False

        # Set the current step to the starting node of the graph
        self.current_node = self.network["starting_node"]  # self.G[0]['starting_node']
        logging.info(f"NETWORK {self.id} \n")
        logging.info(
            f"INIT: Reward balance {self.reward_balance}, n_steps done {self.step_counter}"
        )

    def step(self, action):
        # Execute one time step within the environment
        # self._take_action(action)
        self.source_node = action["source_num"]  # OR with _id alternatively
        self.reward_balance += action["reward"]
        self.current_node = action["target_num"]
        self.step_counter += 1
        node = self.nodes[self.current_node]
        level = node["level"]
        self.max_level = max(self.max_level, level)

        if self.step_counter == self.MAX_STEP:  # 8:
            self.is_done = True

        return {
            "source_node": self.source_node,
            "current_node": self.current_node,
            "reward": action["reward"],
            "total_reward": self.reward_balance,
            "level": level,
            "max_level": self.max_level,
            "n_steps": self.step_counter,
            "done": self.is_done,
        }

    def get_state(self):
        """
        this function returns the current state of the environment.
        State information given by this funciton is less detailed compared
        to the observation.
        """
        return {
            "current_node": self.current_node,
            "total_reward": self.reward_balance,
            "n_steps": self.step_counter,
            "done": self.is_done,
        }

    def observe(self):
        """
        this function returns observation from the environment
        """
        return {
            "current_node": self.current_node,
            "actions_available": [
                n for n in self.action_space if n["source_num"] == self.current_node
            ],
            "next_possible_nodes": np.asarray(
                [
                    n["target_num"]
                    for n in self.action_space
                    if n["source_num"] == self.current_node
                ]
            ),
            "next_possible_rewards": np.asarray(
                [
                    n["reward"]
                    for n in self.action_space
                    if n["source_num"] == self.current_node
                ]
            ),
            "total_reward": self.reward_balance,
            "n_steps": self.step_counter,
            "done": self.is_done,
        }
