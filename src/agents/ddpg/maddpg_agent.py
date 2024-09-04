import torch
from torch import optim
import torch.nn.functional as F

from src.agents.ddpg.iddpg_agent import IddpgAgent
from src.agents.ddpg.ddpg_agent import DdpgAgent
from src.common.replay_buffer import ReplayBuffer
from src.networks.deterministic_actor import DeterministicActor
from src.networks.value_critic import ValueCritic


class MaddpgAgent(IddpgAgent):
    def __init__(self, n_agents, obs_dim, action_dim, hidden_dim, lr, gamma, eps_clip, K_epochs, buffer_size, batch_size, noise_scale: float):
        super().__init__(n_agents, obs_dim, action_dim, hidden_dim, lr, gamma, eps_clip, K_epochs, buffer_size, batch_size, noise_scale)
        self.ddpg_agents = []
        self.memories = []
        global_obs_dim = obs_dim * n_agents
        global_action_dim = action_dim * n_agents
        self.centralized_critic = ValueCritic(obs_dim=global_obs_dim, action_dim=global_action_dim, hidden_dim=hidden_dim)
        self.centralized_critic_optimizer = optim.Adam(self.centralized_critic.parameters(), lr=lr)

        self.centralized_target_critic = ValueCritic(obs_dim=global_obs_dim, action_dim=global_action_dim, hidden_dim=hidden_dim)

        for _ in range(n_agents):
            actor = DeterministicActor(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)
            target_actor = DeterministicActor(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)
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

    def update_centralized_critic(self, global_old_observations, global_old_actions, rewards, next_observations, dones,
                                  gamma):

        predicted_q_values = self.centralized_critic(global_old_observations, global_old_actions)

        next_actions = []
        for agent, next_obs in zip(self.ddpg_agents, next_observations):
            next_action = agent.target_actor(next_obs)
            next_actions.append(next_action)
        global_next_actions = torch.cat(next_actions, dim=-1)

        combined_next_input = torch.cat([next_observations.view(next_observations.size(0), -1), global_next_actions],
                                        dim=-1)
        target_q_values = self.centralized_target_critic(combined_next_input).detach()

        target_q_values = rewards + gamma * (1 - dones) * target_q_values

        critic_loss = F.mse_loss(predicted_q_values, target_q_values)

        self.centralized_critic_optimizer.zero_grad()
        critic_loss.backward()
        self.centralized_critic_optimizer.step()

        return predicted_q_values

    def soft_update(self, target, source, tau=0.001):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def update(self):
        global_rewards, global_old_observations, global_old_actions, global_next_observations, global_dones = [], [], [], [], []

        for idx, agent in enumerate(self.ddpg_agents):
            states, actions, rewards, next_states, dones = agent.get_update_data()

            if rewards is None:
                continue

            global_rewards.append(rewards)
            global_old_observations.append(states)
            global_old_actions.append(actions)
            global_next_observations.append(next_states)
            global_dones.append(dones)

        if len(global_rewards) == 0:
            return

        global_old_observations = torch.stack(global_old_observations).view(-1, self.n_agents * self.obs_dim)
        global_old_actions = torch.stack(global_old_actions).view(-1, self.n_agents * self.action_dim)
        global_rewards = torch.stack(global_rewards).sum(dim=0)
        global_next_observations = torch.stack(global_next_observations).view(-1, self.n_agents * self.obs_dim)
        global_dones = torch.stack(global_dones).sum(dim=0)

        global_observation_values = self.update_centralized_critic(
            global_old_observations=global_old_observations,
            global_old_actions=global_old_actions,
            rewards=global_rewards,
            next_observations=global_next_observations,
            dones=global_dones,
            gamma=self.gamma
        )

        for idx, agent in enumerate(self.ddpg_agents):
            rewards, old_observations, old_actions, next_observations, dones = agent.get_update_data()

            if rewards is None:
                continue

            agent.update_actor(
                observations=old_observations,
            )

        self.soft_update(self.centralized_target_critic, self.centralized_critic)
        for agent in self.ddpg_agents:
            agent.soft_update_critic(tau=0.001)


