import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agents.q.dqn_agent import DqnAgent
from src.agents.q.idqn_agent import IdqnAgent
from src.common.replay_buffer import ReplayBuffer
from src.networks.deep_q_network import DeepQNetwork
from src.networks.mixing_network import MixingNetwork


class QmixAgent(IdqnAgent):
    def __init__(self, n_agents, mixing_hidden_dim, mixing_obs_dim,
                 q_agent_obs_dim, hidden_dim, hidden_output_dim, action_dim,
                 learning_rate, epsilon, gamma, buffer_capacity, batch_size):
        super().__init__(n_agents, q_agent_obs_dim, hidden_dim, hidden_output_dim, action_dim,
                         learning_rate, epsilon, gamma, buffer_capacity, batch_size)

        self.mixing_network = MixingNetwork(
            n_agents=n_agents,
            obs_dim=mixing_obs_dim,
            hidden_dim=mixing_hidden_dim
        )

        self.optimizer = None
        params = []
        self.agents = []
        for _ in range(n_agents):
            q_network = DeepQNetwork(q_agent_obs_dim, hidden_dim, hidden_output_dim, action_dim)
            replay_buffer = ReplayBuffer(batch_size=batch_size, buffer_capacity=buffer_capacity)
            agent = DqnAgent(
                q_network=q_network,
                target_q_network=None,
                optimizer=self.optimizer,
                replay_buffer=replay_buffer,
                epsilon=epsilon,
                gamma=gamma,
                action_dim=action_dim
            )
            self.agents.append(agent)
            params.append(q_network.parameters())

        self.optimizer = torch.optim.Adam(params=itertools.chain(*params), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(batch_size=batch_size, buffer_capacity=buffer_capacity)

        self.batch_size = batch_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.action_dim = action_dim
        self.n_agents = n_agents

    def update(self):
        if self.replay_buffer.can_sample():
            observations, next_observations, actions, action_qs, rewards, dones = self.get_batch()

            q_values_batch, next_q_values_batch = [], []
            for i in range(len(observations)):
                observation, next_state = observations[i], next_observations[i]
                q_values, next_q_values = [], []
                for id in range(len(observation)):
                    action_q = self.agents[id].max_action_q_value(observation=observation[id])
                    next_action_q = self.agents[id].max_action_q_value(observation=next_state[id])
                    q_values.append(action_q)
                    next_q_values.append(next_action_q)

                q_values_batch.append(torch.tensor(q_values, dtype=torch.float32, requires_grad=True))
                next_q_values_batch.append(torch.tensor(next_q_values, dtype=torch.float32, requires_grad=True))

            q_values_batch = torch.stack(q_values_batch).reshape(self.batch_size, -1)
            next_q_values_batch = torch.stack(next_q_values_batch).reshape(self.batch_size, -1)

            state_batch = observations.reshape(self.batch_size, -1)
            next_state_batch = next_observations.reshape(self.batch_size, -1)

            global_q_value = self.mixing_network(q_values_batch, state_batch)
            next_global_q_value = self.mixing_network(next_q_values_batch, next_state_batch)

            rewards = rewards.sum(dim=1, keepdim=True)
            dones = dones.float().sum(dim=1, keepdim=True)

            with torch.no_grad():
                y_tot = rewards + self.gamma * (1 - dones) * next_global_q_value

            loss = F.mse_loss(y_tot, global_q_value)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            return loss.item()
