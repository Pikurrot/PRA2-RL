from collections import defaultdict
import random
from typing import List, DefaultDict, Tuple
import itertools

import numpy as np
from gymnasium.spaces import Space
from gymnasium.spaces.utils import flatdim


class CQL:
    """
    Agent using the Central Q-Learning algorithm
    """
    def __init__(
        self,
        num_agents: int,
        action_spaces: List[Space],
        gamma: float,
        learning_rate: float = 0.5,
        epsilon: float = 1.0,
        **kwargs,
    ):
        """
        Constructor of CQL

        Initializes variables for central Q-learning agent

        :param num_agents (int): number of agents
        :param action_spaces (List[Space]): action spaces of the environment for each agent
        :param gamma (float): discount factor (gamma)
        :param learning_rate (float): learning rate for Q-learning updates
        :param epsilon (float): epsilon value for the central agent
        """
        self.num_agents = num_agents
        self.action_spaces = action_spaces
        self.n_acts = [flatdim(action_space) for action_space in action_spaces]

        self.gamma: float = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        # joint action space
        self.joint_actions = list(itertools.product(*[range(n) for n in self.n_acts]))
        self.num_joint_actions = len(self.joint_actions)

        # Central Q-table. Maps (joint_obs, joint_action_idx) -> Q-value
        self.q_table: DefaultDict = defaultdict(lambda: 0)

    def act(self, obss) -> List[int]:
        if random.random() < self.epsilon:
			# Random action
            joint_action_idx = random.randrange(self.num_joint_actions)
        else:
			# Greedy action
            q_values = [self.q_table[str((tuple(obss), a_idx))] for a_idx in range(self.num_joint_actions)]
            joint_action_idx = np.argmax(q_values)

        return list(self.joint_actions[joint_action_idx])

    def learn(
        self,
        obss: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        n_obss: List[np.ndarray],
        done: bool,
    ):
        """
        Updates the central Q-table based on the collective experience
        """
        joint_obs = tuple(obss)
        joint_n_obs = tuple(n_obss)
        
        # Find the joint action index
        joint_action = tuple(actions)
        joint_action_idx = self.joint_actions.index(joint_action)
        
        key = str((joint_obs, joint_action_idx))
        current_q = self.q_table[key]

        # Optimize for the sum of rewards
        total_reward = sum(rewards)

        if done:
            max_next_q = 0
        else:
            max_next_q = max([self.q_table[str((joint_n_obs, a_idx))] for a_idx in range(self.num_joint_actions)])

        target = total_reward + self.gamma * max_next_q
        self.q_table[key] += self.learning_rate * (target - current_q)

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """
        Updates the hyperparameters
        """
        self.epsilon = 1.0 - (min(1.0, timestep / (0.8 * max_timestep))) * 0.99

