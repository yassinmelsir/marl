import torch
from torch import optim

from src.agents.ib_agent import IbAgent
from src.agents.q.dqn_agent import DqnAgent
from src.common.replay_buffer import ReplayBuffer
from src.networks.deep_q_network import DeepQNetwork


class IdqnAgent(IbAgent):
    def __init__(self, buffer_capacity, batch_size, n_agents, action_dim, obs_dim, hidden_dim, hidden_output_dim, learning_rate, epsilon, gamma):
        super().__init__(buffer_capacity, batch_size, action_dim, obs_dim, learning_rate, epsilon, gamma)
        for _ in range(n_agents):
            q_network = DeepQNetwork(obs_dim, hidden_dim, hidden_output_dim, action_dim)
            target_q_network = DeepQNetwork(obs_dim, hidden_dim, hidden_output_dim, action_dim)
            target_q_network.load_state_dict(q_network.state_dict())
            optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
            replay_buffer = ReplayBuffer(batch_size=batch_size, buffer_capacity=buffer_capacity)
            agent = DqnAgent(
                q_network=q_network,
                target_q_network=target_q_network,
                optimizer=optimizer,
                replay_buffer=replay_buffer,
                epsilon=epsilon,
                gamma=gamma,
                action_dim=action_dim
            )
            self.agents.append(agent)