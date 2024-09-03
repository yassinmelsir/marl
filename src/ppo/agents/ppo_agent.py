import torch
from torch import optim, nn
import torch.nn.functional as F


class PpoAgent:
    def __init__(self, actor: nn.Module, critic: nn.Module, lr: float, gamma: float, eps_clip: float, K_epochs: int):
        self.actor = actor
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = critic
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

    def select_action(self, observation):
        with torch.no_grad():
            action_probs = self.actor(observation)
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

        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        old_observations = torch.stack([torch.FloatTensor(row) for row in memory.observations])
        old_actions = torch.tensor(memory.actions, dtype=torch.long)
        old_log_probs = torch.tensor(memory.log_probs, dtype=torch.float32)

        for _ in range(self.K_epochs):
            observation_values = self.critic(old_observations)
            advantages = rewards - observation_values.detach()

            critic_loss = 0.5 * F.mse_loss(observation_values, rewards)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            log_probs = self.actor(old_observations)
            dist = torch.distributions.Categorical(log_probs)
            new_log_probs = dist.log_prob(old_actions)

            ratios = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

