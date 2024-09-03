import numpy as np
import torch
from torch import optim
import torch.nn.functional as F

from src.indep.ppo.actor_critic import ActorCritic


class PpoAgent:
    def __init__(self, obs_dim, action_dim, lr, gamma, eps_clip, K_epochs):
        self.actor_critic = ActorCritic(obs_dim=obs_dim, action_dim=action_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

    def select_action(self, observation):
        with torch.no_grad():
            action_probs, _ = self.actor_critic(observation)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action)

    def update(self, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.dones)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.stack([torch.FloatTensor(row) for row in memory.rewards])
        old_observations = torch.stack([torch.FloatTensor(row) for row in memory.observations])
        old_actions = torch.stack([torch.IntTensor(row) for row in memory.actions ])
        old_log_probs = torch.stack([torch.FloatTensor(row) for row in memory.log_probs])

        for _ in range(self.K_epochs):
            log_probs, observation_values = self.actor_critic(old_observations)
            dist = torch.distributions.Categorical(log_probs)
            new_log_probs = dist.log_prob(old_actions)

            ratios = torch.exp(new_log_probs - old_log_probs)
            advantages = rewards - observation_values.detach().squeeze()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * F.mse_loss(observation_values.squeeze(), rewards)

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

