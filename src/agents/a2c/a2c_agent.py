from typing import Union

import torch
from torch import optim, nn
import torch.nn.functional as F

from src.common.memory import Memory


class A2cAgent:
    def __init__(self, actor: nn.Module, critic: nn.Module, memory: Memory, lr: float, gamma: float, eps_clip: float, K_epochs: int, entropy_coefficient: float):
        self.actor = actor
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = critic
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.memory = memory

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_coefficient = entropy_coefficient

    def select_action(self, observation):
        with torch.no_grad():
            action_probs = self.actor(observation)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action)

    def get_update_data(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.dones)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        old_observations = torch.stack([torch.FloatTensor(row) for row in self.memory.observations])
        next_observations = torch.stack([torch.FloatTensor(row) for row in self.memory.next_observations])
        old_actions = torch.tensor(self.memory.actions, dtype=torch.long)

        return rewards, old_observations, old_actions, next_observations

    def update_actor(self, old_observations, old_actions, rewards, observation_values, next_observation_values):

        advantages = rewards + self.gamma * next_observation_values - observation_values.detach()

        log_probs = self.actor(old_observations)
        dist = torch.distributions.Categorical(log_probs)
        new_log_probs = dist.log_prob(old_actions)

        if self.entropy_coefficient is not None:
            entropy = dist.entropy().mean()  # Encourage exploration
            actor_loss = -torch.mean(new_log_probs * advantages) - self.entropy_coefficient * entropy
        else:
            actor_loss = -torch.mean(new_log_probs * advantages)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def update_critic(self, old_observations, rewards):
        observation_values = self.critic(old_observations)
        critic_loss = 0.5 * F.mse_loss(observation_values, rewards)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return observation_values

    def update(self):
        rewards, old_observations, old_actions, next_observations = self.get_update_data()

        for _ in range(self.K_epochs):
            observation_values = self.update_critic(
                old_observations=old_observations,
                rewards=rewards
            )

            next_observation_values = self.critic(next_observations)

            self.update_actor(
                old_observations=old_observations,
                old_actions=old_actions,
                rewards=rewards,
                observation_values=observation_values,
                next_observation_values=next_observation_values
            )

