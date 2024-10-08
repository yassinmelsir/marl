import torch
from torch import optim
import torch.nn.functional as F

from src.agents.ddpg.ddpg_agent import DdpgAgent
from src.agents.ddpg.iddpg_agent import IddpgAgent
from src.common.replay_buffer import ReplayBuffer
from src.networks.gumbel_actor import GumbelActor
from src.networks.value_critic import ValueCritic


class MaddpgAgent(IddpgAgent):
    def __init__(self, n_agents, obs_dim, action_dim, hidden_dim, learning_rate, gamma, epsilon, K_epochs, buffer_capacity,
                         batch_size, noise_scale, temperature):
        super().__init__(n_agents, obs_dim, action_dim, hidden_dim, learning_rate, gamma, epsilon, K_epochs, buffer_capacity,
                         batch_size, noise_scale, temperature)
        self.agents = []
        global_obs_dim = obs_dim * n_agents
        global_action_dim = action_dim * n_agents
        self.centralized_critic = ValueCritic(obs_dim=global_obs_dim, action_dim=global_action_dim,
                                              hidden_dim=hidden_dim)
        self.centralized_critic_optimizer = optim.Adam(self.centralized_critic.parameters(), lr=learning_rate)

        self.centralized_target_critic = ValueCritic(obs_dim=global_obs_dim, action_dim=global_action_dim,
                                                     hidden_dim=hidden_dim)

        for _ in range(n_agents):
            actor = GumbelActor(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)
            target_actor = GumbelActor(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)
            replay_buffer = ReplayBuffer(buffer_capacity=buffer_capacity, batch_size=batch_size)
            agent = DdpgAgent(
                actor=actor,
                critic=self.centralized_critic,
                target_actor=target_actor,
                target_critic=self.centralized_target_critic,
                replay_buffer=replay_buffer,
                learning_rate=learning_rate,
                gamma=gamma,
                epsilon=epsilon,
                K_epochs=K_epochs,
                noise_scale=noise_scale,
            )
            self.agents.append(agent)

        self.gamma = gamma
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def reshaped_batch_item_by_agent(self, batch, dim):
        return batch.view(batch.size(0), self.n_agents, dim)

    def cat_batch_item_to_global(self, batch, dim):
        return batch.view(-1, self.n_agents * dim)

    def get_action_probs(self, obs):
        action_probs = []
        for idx, agent in enumerate(self.agents):
            agent_obs = obs[:, idx, :]
            action_ps = agent.target_actor(agent_obs)
            action_probs.append(action_ps)

        return torch.stack(action_probs)

    def get_predicted_q_values(self, observations, action_probs):
        cat_obs = self.cat_batch_item_to_global(observations, dim=self.obs_dim)
        cat_action_probs = self.cat_batch_item_to_global(action_probs, dim=self.action_dim)

        return self.centralized_critic(cat_obs, cat_action_probs)

    def update_centralized_critic(self, observations, next_observations, action_probs, rewards, dones):

        predicted_q_values = self.get_predicted_q_values(observations=observations, action_probs=action_probs)

        next_action_probs = self.get_action_probs(obs=next_observations)
        cat_next_obs = self.cat_batch_item_to_global(next_observations, dim=self.obs_dim)
        cat_action_probs = self.cat_batch_item_to_global(next_action_probs, dim=self.action_dim)

        target_q_values = self.centralized_target_critic(cat_next_obs, cat_action_probs)
        rewards = rewards.sum(dim=1, keepdim=True)
        dones = dones.float().sum(dim=1, keepdim=True)

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

        _ = self.update_centralized_critic(
            observations=global_obs,
            next_observations=global_next_obs,
            action_probs=global_action_probs,
            rewards=global_rewards,
            dones=global_dones,
        )

        for idx, agent in enumerate(self.agents):
            predicted_q_values = self.get_predicted_q_values(observations=global_obs, action_probs=global_action_probs)
            actor_loss = -predicted_q_values.mean()
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()

        self.soft_update(self.centralized_target_critic, self.centralized_critic)
        for agent in self.agents:
            agent.soft_update_critic(tau=0.001)
