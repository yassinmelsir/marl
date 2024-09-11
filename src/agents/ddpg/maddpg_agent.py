from typing import Union

import torch
from torch import optim
import torch.nn.functional as F

from src.agents.ddpg.iddpg_agent import IddpgAgent
from src.agents.ddpg.ddpg_agent import DdpgAgent
from src.common.replay_buffer import ReplayBuffer
from src.networks.gumbel_actor import GumbelActor
from src.networks.value_critic import ValueCritic


class MaddpgAgent(IddpgAgent):
    def __init__(self, n_agents: int, obs_dim: int, action_dim: int, hidden_dim: int, lr: float, gamma: float, eps_clip: float, K_epochs: int, buffer_size: int, batch_size: int, noise_scale: Union[float, None], temperature: float):
        super().__init__(n_agents, obs_dim, action_dim, hidden_dim, lr, gamma, eps_clip, K_epochs, buffer_size, batch_size, noise_scale,  temperature)
        self.ddpg_agents = []
        global_obs_dim = obs_dim * n_agents
        global_action_dim = action_dim * n_agents
        self.centralized_critic = ValueCritic(obs_dim=global_obs_dim, action_dim=global_action_dim, hidden_dim=hidden_dim)
        self.centralized_critic_optimizer = optim.Adam(self.centralized_critic.parameters(), lr=lr)

        self.centralized_target_critic = ValueCritic(obs_dim=global_obs_dim, action_dim=global_action_dim, hidden_dim=hidden_dim)

        for _ in range(n_agents):
            actor = GumbelActor(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)
            target_actor = GumbelActor(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)
            replay_buffer = ReplayBuffer(buffer_size=buffer_size, batch_size=batch_size)
            ddpg_agent = DdpgAgent(
                actor=actor,
                critic=self.centralized_critic,
                target_actor=target_actor,
                target_critic=self.centralized_target_critic,
                replay_buffer=replay_buffer,
                lr=lr,
                gamma=gamma,
                eps_clip=eps_clip,
                K_epochs=K_epochs,
                noise_scale=noise_scale,
            )
            self.ddpg_agents.append(ddpg_agent)

        self.gamma = gamma
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def update_centralized_critic(self, observations, next_observations, action_probs, rewards, dones):

        predicted_q_values = self.centralized_critic(observations, action_probs)


        next_actions = []
        reshaped_next_obs = next_observations.view(next_observations.size(0), self.n_agents, self.obs_dim)
        for idx, agent in enumerate(self.agents):
            agent_next_obs = reshaped_next_obs[:, idx, :]
            next_action = agent.target_actor(agent_next_obs)
            next_actions.append(next_action)

        global_next_actions = torch.cat(next_actions, dim=-1)

        combined_next_input = torch.cat([next_observations.view(next_observations.size(0), -1), global_next_actions],
                                        dim=-1)
        target_q_values = self.centralized_target_critic(combined_next_input).detach()

        target_q_values = rewards + self.gamma * (1 - dones) * target_q_values

        critic_loss = F.mse_loss(predicted_q_values, target_q_values)

        self.centralized_critic_optimizer.zero_grad()
        critic_loss.backward()
        self.centralized_critic_optimizer.step()

        return predicted_q_values

    def soft_update(self, target, source, tau=0.001):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def update(self):
        if not self.replay_buffer.can_sample():
            return None

        global_obs, global_next_obs, global_actions, global_action_probs, global_rewards, global_dones = self.get_batch()

        global_obs = global_obs.view(-1, self.n_agents * self.obs_dim)
        global_next_obs = global_next_obs.view(-1, self.n_agents * self.obs_dim)

        global_action_probs = global_action_probs.view(-1, self.n_agents * self.action_dim)
        global_rewards = global_rewards.sum(dim=0)
        global_dones = global_dones.sum(dim=0)

        global_observation_values = self.update_centralized_critic(
            observations=global_obs,
            next_observations=global_next_obs,
            action_probs=global_action_probs,
            rewards=global_rewards,
            dones=global_dones,
        )

        reshaped_global_obs = global_obs.view(global_obs.size(0), self.n_agents, self.obs_dim)
        for idx, agent in enumerate(self.ddpg_agents):
            agent_obs = reshaped_global_obs[:, idx, :]
            agent.update_actor(
                observations=agent_obs,
            )

        self.soft_update(self.centralized_target_critic, self.centralized_critic)
        for agent in self.ddpg_agents:
            agent.soft_update_critic(tau=0.001)


