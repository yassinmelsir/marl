from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import torch

from pettingzoo.mpe import simple_spread_v3


@dataclass
class SimpleSpreadParams:
    n: int
    local_ratio: float
    max_cycles: int


class SimpleSpread:
    def __init__(self, params: SimpleSpreadParams):
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

    def last(self) -> tuple[Any, Any, Any, Any]:
        return self.env.last()

    def select_action(self, action):
        self.env.step(action)

    def observe_agent(self, agent_id) -> Any:
        return self.env.observe(agent_id)

    def step(self, agent):
        observations = []
        next_observations = []
        rewards = []
        dones = []
        actions = []
        action_probs = []

        global_experience = (
            observations,
            next_observations,
            actions,
            action_probs,
            rewards,
            dones
        )

        for idx, agent_id in enumerate(self.env.agents):
            observation, reward, termination, truncation, _ = self.env.last()
            obs_tensor = torch.FloatTensor(observation)

            if termination or truncation:
                return rewards, [True]
            else:
                action, action_probs_tensor = agent.agents[idx].select_action(observation=obs_tensor)

            self.env.step(action)
            next_observation = self.env.observe(agent_id)
            done = termination or truncation

            agent_experience = (
                obs_tensor,
                next_observation,
                action,
                action_probs_tensor,
                reward,
                done
            )

            agent.save_agent_data(global_experience, agent_experience, agent=agent.agents[idx])

        rewards, dones = agent.save_global_data(global_experience)

        return rewards, dones