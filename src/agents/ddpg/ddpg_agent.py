from typing import Union

import numpy as np
import torch
from torch import optim, nn
import torch.nn.functional as F

from src.common.replay_buffer import ReplayBuffer


class DdpgAgent:
    def __init__(self, actor: nn.Module, critic: nn.Module, target_actor: nn.Module, target_critic: nn.Module,
                 replay_buffer: ReplayBuffer, lr: float, gamma: float, eps_clip: float, K_epochs: int, noise_scale: Union[float, None]):
        self.actor = actor
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = critic
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.target_actor = target_actor
        self.target_critic = target_critic

        self.replay_buffer = replay_buffer

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.noise_scale = noise_scale

    def select_action(self, observation):
        with torch.no_grad():
            action = self.actor(observation).squeeze(0).cpu().numpy()

        if self.noise_scale is not None:
            noise = self.noise_scale * np.random.randn(*action.shape)
            action = action + noise

        action = np.clip(action, -1, 1)

        return action

    def get_update_data(self):
        if not self.replay_buffer.can_sample():
            return None, None, None, None, None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        return states, actions, rewards, next_states, dones

    def update_actor(self, observations):
        predicted_actions = self.actor(observations)

        actor_loss = -self.critic(observations, predicted_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def update_critic(self, state, action, reward, next_state, done):
        predicted_q_value = self.critic(state, action)

        next_action = self.target_actor(next_state)

        target_q_value = self.target_critic(next_state, next_action).detach()

        target_q_value = reward + self.gamma * (1 - done) * target_q_value

        critic_loss = F.mse_loss(predicted_q_value, target_q_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss

    def soft_update_critic(self, tau):
        self.soft_update(tau=tau, target_network=self.target_critic, source_network=self.critic)

    def soft_update_actor(self, tau):
        self.soft_update(tau=tau, target_network=self.target_actor, source_network=self.actor)

    @staticmethod
    def soft_update(tau, target_network, source_network):
        for target_param, param in zip(target_network.parameters(), source_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def update(self):
        states, actions, rewards, next_states, dones = self.get_update_data()

        for _ in range(self.K_epochs):
            critic_loss = self.update_critic(
                states, actions, rewards, next_states, dones
            )
            self.update_actor(states)
            self.soft_update_critic(tau=0.001)
            self.soft_update_actor(tau=0.001)

