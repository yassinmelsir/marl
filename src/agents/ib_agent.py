from typing import Union

import torch
from src.agents.ddpg.ddpg_agent import DdpgAgent
from src.common.replay_buffer import ReplayBuffer
from src.networks.gumbel_actor import GumbelActor
from src.networks.value_critic import ValueCritic


class IbAgent:
    def __init__(self, buffer_capacity: int, batch_size: int, action_dim: int, obs_dim: int, learning_rate: float, epsilon: float, gamma: float):
            self.agents = []
            self.replay_buffer = ReplayBuffer(batch_size=batch_size, buffer_capacity=buffer_capacity)
            self.epsilon = epsilon
            self.gamma = gamma
            self.obs_dim = obs_dim
            self.action_dim = action_dim
            self.batch_size = batch_size
            self.learning_rate = learning_rate

    def get_batch(self):

        batch = self.replay_buffer.sample()
        observations, next_observations, actions, action_probs, rewards, dones = zip(*batch)

        return (
            torch.stack(observations),
            torch.stack(next_observations),
            torch.stack(actions),
            torch.stack(action_probs),
            torch.stack(rewards),
            torch.stack(dones)
        )

    def step(self, env):
        observations = []
        next_states = []
        rewards = []
        dones = []
        actions = []
        action_probs = []

        for idx, agent_id in enumerate(env.agents):
            observation, reward, termination, truncation, _ = env.last()
            obs_tensor = torch.FloatTensor(observation)

            if termination or truncation:
                return rewards, [True]
            else:
                action_tuple = self.agents[idx].select_action(observation=obs_tensor)
                action, action_probs_tensor = action_tuple

            env.step(action)
            next_observation = env.observe(agent_id)

            next_obs_tensor = torch.FloatTensor(next_observation)
            action_tensor = torch.IntTensor([action])
            done_tensor = torch.BoolTensor([termination or truncation])
            reward_tensor = torch.FloatTensor([reward])

            observations.append(obs_tensor)
            next_states.append(next_obs_tensor)
            actions.append(action_tensor)
            action_probs.append(action_probs_tensor)
            rewards.append(reward_tensor)
            dones.append(done_tensor)

            experience = (
                obs_tensor,
                next_obs_tensor,
                action_tensor,
                action_probs_tensor,
                reward_tensor,
                done_tensor
            )

            self.agents[idx].replay_buffer.add(experience)

        observations = torch.stack(observations)
        next_states = torch.stack(next_states)
        actions = torch.stack(actions)
        action_probs = torch.stack(action_probs)
        rewards = torch.tensor(rewards)
        dones = torch.tensor(dones)

        experience = (
            observations,
            next_states,
            actions,
            action_probs,
            rewards,
            dones
        )

        self.replay_buffer.add(experience)

        return rewards, dones

    def update(self):
        for idx, agent in enumerate(self.agents):
            agent.update()