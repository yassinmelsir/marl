import torch
from torch import optim
import torch.nn.functional as F
from src.agents.ddpg.ddpg_agent import DdpgAgent
from src.agents.ddpg.iddpg_agent import IddpgAgent
from src.common.replay_buffer import ReplayBuffer
from src.networks.gumbel_actor import GumbelActor
from src.networks.value_critic import ValueCritic
import numpy as np


class MaddpgAgent(IddpgAgent):
    def __init__(self, agent_params, central_params):
        super().__init__(agent_params=agent_params)
        self.agents = []

        self.gamma = central_params.gamma
        self.n_agents = len(agent_params)
        self.batch_size = central_params.batch_size

        self.obs_dim = central_params.obs_dim
        self.action_dim = central_params.action_dim
        self.central_obs_dim = (central_params.obs_dim + self.action_dim) * self.n_agents

        self.full_length_srcs = central_params.full_length_srcs

        self.start_token = None

        self.transformer = central_params.transformer

        if self.transformer is not None:
            self.t_dim = self.transformer.transformer.d_model
            self.central_obs_dim += self.t_dim
            self.start_token = torch.zeros(central_params.batch_size, 1, self.t_dim)

        self.centralized_critic = ValueCritic(obs_dim=self.central_obs_dim,
                                              hidden_dim=central_params.hidden_dim)
        self.centralized_critic_optimizer = optim.Adam(self.centralized_critic.parameters(),
                                                       lr=central_params.learning_rate)

        self.centralized_target_critic = ValueCritic(obs_dim=self.central_obs_dim,
                                                     hidden_dim=central_params.hidden_dim)

        for param in agent_params:
            actor = GumbelActor(obs_dim=param.obs_dim, action_dim=param.action_dim, hidden_dim=param.hidden_dim)
            target_actor = GumbelActor(obs_dim=param.obs_dim, action_dim=param.action_dim, hidden_dim=param.hidden_dim)
            replay_buffer = ReplayBuffer(buffer_capacity=param.buffer_capacity, batch_size=param.batch_size)
            agent = DdpgAgent(
                actor=actor,
                critic=self.centralized_critic,
                target_actor=target_actor,
                target_critic=self.centralized_target_critic,
                replay_buffer=replay_buffer,
                learning_rate=param.learning_rate,
                gamma=param.gamma,
                epsilon=param.epsilon,
                noise_scale=param.noise_scale,
                K_epochs=param.K_epochs
            )
            self.agents.append(agent)

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

    def get_obs(self, observations, action_probs, indices):
        obs = self.cat_batch_item_to_global(observations, dim=self.obs_dim)
        action_probs = self.cat_batch_item_to_global(action_probs, dim=self.action_dim)

        if self.transformer is not None:
            buffer = list(self.replay_buffer.buffer)
            if self.full_length_srcs:
                prev_timesteps = buffer[:max(indices)]
            else:
                prev_timesteps = [buffer[i] for i in indices]

            for i in range(len(prev_timesteps)):
                prev_timesteps[i] = list(prev_timesteps[i])
                for j in range(len(prev_timesteps[i])):
                    prev_timesteps[i][j] = prev_timesteps[i][j].view(1, -1)

                features = [feature for index, feature in enumerate(prev_timesteps[i]) if index in [0, 1]] # keep observations, actions
                prev_timesteps[i] = torch.cat([torch.cat(features, dim=1), torch.zeros(1, 1)], dim=1)

            prev_timesteps = torch.stack(prev_timesteps)
            src = [prev_timesteps[:i] for i in indices]

            trts = [self.generate_next_sequence(s) for s in src]

            critic_obs = []

            for i in range(obs.shape[0]):
                critic_obs.append(torch.cat((trts[i][0].view(-1), obs[i], action_probs[i])))

            return torch.stack(critic_obs)
        else:
            return torch.cat((obs, action_probs), dim=1)


    def generate_next_sequence(self, observations):
        return self.transformer(observations, self.start_token)

    def update_centralized_critic(
            self,
            observations,
            next_observations,
            rewards,
            dones
    ):
        predicted_q_values = self.centralized_critic(observations)
        target_q_values = self.centralized_critic(next_observations)

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

        global_obs, global_next_obs, global_actions, global_action_probs, global_rewards, global_dones, indices = self.get_batch()

        global_obs = self.get_obs(
            observations=global_obs,
            action_probs=global_action_probs,
            indices=indices
        )

        global_next_action_probs = self.get_action_probs(obs=global_next_obs)
        global_next_obs = self.get_obs(
            observations=global_next_obs,
            action_probs=global_next_action_probs,
            indices=indices
        )

        _ = self.update_centralized_critic(
            observations=global_obs,
            next_observations=global_next_obs,
            rewards=global_rewards,
            dones=global_dones,
        )


        for idx, agent in enumerate(self.agents):
            predicted_q_values = self.centralized_critic(global_obs)
            actor_loss = -predicted_q_values.mean()
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()

        self.soft_update(self.centralized_target_critic, self.centralized_critic)
        for agent in self.agents:
            agent.soft_update_critic(tau=0.001)
