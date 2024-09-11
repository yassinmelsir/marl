from typing import Union

import numpy as np
import torch
from torch import optim, nn
import torch.nn.functional as F

from src.common.replay_buffer import ReplayBuffer


class DdpgAgent:
    def __init__(self, actor: nn.Module, critic: nn.Module, target_actor: nn.Module, target_critic: nn.Module,
                 replay_buffer: ReplayBuffer, learning_rate: float, gamma: float, epsilon: float, K_epochs: int, noise_scale: Union[float, None]):
        self.actor = actor
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)

        self.critic = critic
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.target_actor = target_actor
        self.target_critic = target_critic

        self.replay_buffer = replay_buffer

        self.gamma = gamma
        self.epsilon = epsilon
        self.K_epochs = K_epochs
        self.noise_scale = noise_scale

    def select_action(self, observation):
        with torch.no_grad():
            logits = self.actor(observation).squeeze(0)
            action_probs = self.actor.sample_gumbel_softmax(logits)
            action = torch.argmax(action_probs, dim=-1).item()

        return action, action_probs

    def update_actor(self, observations):
        logits = self.actor(observations)
        gumbel_softmax_actions = self.actor.sample_gumbel_softmax(logits)

        actor_loss = -self.critic(observations, gumbel_softmax_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def update_critic(self, observations, actions, rewards, next_states, dones):

        predicted_q_values = self.critic(observations, actions)

        next_actions = self.target_actor(next_states)

        target_q_values = self.target_critic(next_states, next_actions).detach()

        adjusted_target_q_values = rewards + self.gamma * (1 - dones.float()) * target_q_values


        critic_loss = F.mse_loss(predicted_q_values, adjusted_target_q_values)

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

    def get_batch(self):
        batch = self.replay_buffer.sample()
        observations, next_observations, action, action_probs, rewards, dones = zip(*batch)

        return (
            torch.stack(observations),
            torch.stack(next_observations),
            torch.stack(action),
            torch.stack(action_probs),
            torch.stack(rewards),
            torch.stack(dones)
        )

    def update(self):
        if self.replay_buffer.can_sample():

            observations, next_observations, actions, action_probs, rewards, dones = self.get_batch()


            for _ in range(self.K_epochs):
                critic_loss = self.update_critic(
                    observations=observations,
                    actions=action_probs,
                    rewards=rewards,
                    next_states=next_observations,
                    dones=dones
                )
                self.update_actor(observations)
                self.soft_update_critic(tau=0.001)
                self.soft_update_actor(tau=0.001)

