import torch
from torch import optim, nn
import torch.nn.functional as F

from src.common.memory import Memory


class PpoAgent:
    def __init__(self, actor: nn.Module, critic: nn.Module, memory: Memory, learning_rate: float, gamma: float, epsilon: float, K_epochs: int):
        self.actor = actor
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)

        self.critic = critic
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.memory = memory

        self.gamma = gamma
        self.epsilon = epsilon
        self.K_epochs = K_epochs

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
        old_actions = torch.tensor(self.memory.actions, dtype=torch.long)
        old_log_probs = torch.tensor(self.memory.log_probs, dtype=torch.float32)

        return rewards, old_observations, old_actions, old_log_probs

    def update_actor(self, old_observations, old_actions, old_log_probs, rewards, observation_values):

        advantages = rewards - observation_values.detach()
        log_probs = self.actor(old_observations)
        dist = torch.distributions.Categorical(log_probs)
        new_log_probs = dist.log_prob(old_actions)

        ratios = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages

        actor_loss = -torch.min(surr1, surr2).mean()

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
        rewards, old_observations, old_actions, old_log_probs = self.get_update_data()

        for _ in range(self.K_epochs):
            observation_values = self.update_critic(
                old_observations=old_observations,
                rewards=rewards
            )

            self.update_actor(
                old_observations=old_observations,
                old_actions=old_actions,
                old_log_probs=old_log_probs,
                rewards=rewards,
                observation_values=observation_values
            )

