from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import torch

from pettingzoo.mpe import simple_spread_v3



class SumoIngolstadt:
    def __init__(self, params):
        self.env = simple_spread_v3.env(
            N=params.n,
            local_ratio=params.local_ratio,
            max_cycles=params.max_cycles
        )
        self.env.reset()
        self.n_agents = len(self.env.agents)

        obs_size, action_size = self.get_obs_action_size()
        self.obs_size = obs_size
        self.action_size = action_size

    def get_obs_action_size(self):
        first_agent = self.env.agents[0]
        observation_space = self.env.observation_space(first_agent)
        action_space = self.env.action_space(first_agent)

        if isinstance(observation_space, gym.spaces.Box):
            obs_size = observation_space.shape[0]
        else:
            raise ValueError(f"Unexpected observation space type: {type(observation_space)}")

        if isinstance(action_space, gym.spaces.Discrete):
            print(f"action_space: {action_space}")
            action_size = action_space.n
        else:
            raise ValueError(f"Unexpected action space type: {type(action_space)}")

        return obs_size, action_size

    def get_env(self):
        return self.env

    def reset(self):
        self.env.reset()

