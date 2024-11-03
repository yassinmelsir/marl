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

    def get_obs(self, observations, actions, action_probs, rewards, dones, indices):
        cat_obs = self.cat_batch_item_to_global(observations, dim=self.obs_dim)
        cat_action_probs = self.cat_batch_item_to_global(action_probs, dim=self.action_dim)

        if self.full_length_srcs:
            buffer = list(self.replay_buffer.buffer)

            prev_timesteps = buffer[:max(indices)]
            for i in range(len(prev_timesteps)):
                prev_timesteps[i] = list(prev_timesteps[i])
                for j in range(len(prev_timesteps[i])):
                    prev_timesteps[i][j] = prev_timesteps[i][j].view(1, -1)

                features = [feature for index, feature in enumerate(prev_timesteps[i]) if index not in [1, 3]]
                prev_timesteps[i] = torch.cat([torch.cat(features, dim=1), torch.zeros(1, 1)], dim=1)



            prev_timesteps = torch.stack(prev_timesteps)
            src = [prev_timesteps[:i] for i in indices]
        else:


        if self.transformer is not None:
            if self.full_length_srcs:
                src = self.get_srcs(observations, actions, action_probs, rewards, dones, indices)
                trts = [self.generate_next_sequence(s) for s in src]
            else:
                breakpoint()
                src = [torch.cat((cat_obs, actions, rewards, dones)) for i in cat_obs.shape[0]]

            critic_obs = []

            for i in range(cat_obs.shape[0]):
                critic_obs.append(torch.cat((trts[i][0].view(-1), cat_obs[i], cat_action_probs[i])))

            critic_obs = torch.stack(critic_obs)
        else:
            critic_obs = torch.cat((cat_obs, cat_action_probs), dim=1)


        return src

    def get_predicted_q_values(self, observations, actions, action_probs, rewards, dones, indices):

        critic_obs = self.get_obs(observations, actions, action_probs, rewards, dones, indices)

        return self.centralized_critic(critic_obs)

    def get_target_q_values(self, next_observations):
        next_action_probs = self.get_action_probs(obs=next_observations)
        cat_next_obs = self.cat_batch_item_to_global(next_observations, dim=self.obs_dim)
        cat_next_action_probs = self.cat_batch_item_to_global(next_action_probs, dim=self.action_dim)
        critic_input = torch.cat((cat_next_obs, cat_next_action_probs))
        return self.centralized_target_critic(critic_input)

    def generate_next_sequence(self, observations):
        return self.transformer(observations, self.start_token)

    def update_centralized_critic(
            self,
            observations,
            next_observations,
            action_probs,
            rewards,
            dones,
            indices
    ):
        predicted_q_values = self.get_predicted_q_values(
            observations=observations,
            actions=actions,
            action_probs=action_probs,
            indices=indices
        )

        next_action_probs = self.get_action_probs(obs=next_observations)
        target_q_values = self.get_predicted_q_values(
            observations=next_observations,
            action_probs=next_action_probs,
            indices=indices
        )

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



        _ = self.update_centralized_critic(
            observations=global_obs,
            next_observations=global_next_obs,
            action_probs=global_action_probs,
            rewards=global_rewards,
            dones=global_dones,
            indices=indices
        )

        for idx, agent in enumerate(self.agents):
            predicted_q_values = self.get_predicted_q_values(observations=global_obs, action_probs=global_action_probs, indices=indices)
            actor_loss = -predicted_q_values.mean()
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()

        self.soft_update(self.centralized_target_critic, self.centralized_critic)
        for agent in self.agents:
            agent.soft_update_critic(tau=0.001)
