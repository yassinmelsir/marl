import random, torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchrl.data import ReplayBuffer, ListStorage

from value_decomposition.dqn.deep_q_network import DeepQNetwork
from value_decomposition.qmix.q_mixing_network import QMixingNetwork


class QmAgent:
    def __init__(self, n_agents, embed_dim, mixing_state_dim,
                 q_agent_state_dim, hidden_dim, hidden_output_dim, n_actions,
                 learning_rate, epsilon, gamma, buffer_capacity, batch_size):

        self.q_mixing_network = QMixingNetwork(
            n_agents=n_agents,
            state_dim=mixing_state_dim,
            embed_dim=embed_dim
        )

        self.agents = nn.ModuleDict({
            f"agent_{i}": DeepQNetwork(
                state_dim=q_agent_state_dim,
                hidden_dim=hidden_dim,
                hidden_output_dim=hidden_output_dim,
                n_actions=n_actions
            )
            for i in range(n_agents)})

        params = list(self.agents.parameters()) + list(self.q_mixing_network.parameters())
        self.optimizer = torch.optim.Adam(params=params, lr=learning_rate)

        self.replay_buffer = ReplayBuffer(batch_size=buffer_capacity, storage=ListStorage(max_size=buffer_capacity))
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.n_actions = n_actions
        self.q_agent_state_dim = q_agent_state_dim
        self.n_agents = n_agents

    def select_action(self, state, agent_id):
        if torch.rand(1).item() < self.epsilon:
            return torch.randint(0, self.n_actions, (1,)).item()
        else:
            with torch.no_grad():
                q_values, _ = self.agents[agent_id](state.unsqueeze(0))
                return q_values.argmax().item()

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones, global_states, next_global_states = self.get_batch()

        q_values = torch.stack([self.agents[f'agent_{i}'](states[:, i]) for i in range(self.n_agents)], dim=1)
        chosen_q_values = q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)

        joint_q_values = self.q_mixing_network(chosen_q_values, global_states)

        with torch.no_grad():
            next_q_values = torch.stack([self.agents[f'agent_{i}'](next_states[:, i]) for i in range(self.n_agents)], dim=1)
            next_max_q_values = next_q_values.max(dim=2)[0]
            target_max_q_values = self.q_mixing_network(next_max_q_values, next_global_states)
            targets = rewards.sum(dim=1, keepdim=True) + (1 - dones) * self.gamma * target_max_q_values

        loss = F.mse_loss(joint_q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def get_batch(self):
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones, global_states, next_global_states = zip(*batch)

        states = torch.FloatTensor(states).view(self.batch_size, self.n_agents, -1)
        next_states = torch.FloatTensor(next_states).view(self.batch_size, self.n_agents, -1)
        actions = torch.LongTensor(actions).view(self.batch_size, self.n_agents)
        rewards = torch.FloatTensor(rewards).view(self.batch_size, self.n_agents)
        dones = torch.FloatTensor(dones).view(self.batch_size, 1)
        global_states = torch.FloatTensor(global_states).view(self.batch_size, -1)
        next_global_states = torch.FloatTensor(next_global_states).view(self.batch_size, -1)

        return states, actions, rewards, next_states, dones, global_states, next_global_states

    def add_to_buffer(self, data):
        self.replay_buffer.add(data)

    def step(self, env):
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            # Convert scalar observations to 1D numpy arrays
            if np.isscalar(observation):
                observation = np.array([observation])

            states.append(observation)
            rewards.append(reward)
            dones.append(termination or truncation)

            if termination or truncation:
                action = None
            else:
                action = self.select_action(torch.FloatTensor(observation), agent_id)

            actions.append(action)
            env.step(action)

            next_observation = env.observe(agent_id)[0]
            # Convert scalar next_observation to 1D numpy array
            if np.isscalar(next_observation):
                next_observation = np.array([next_observation])
            next_states.append(next_observation)

        # Ensure all states and next_states are 1D arrays
        states = [np.atleast_1d(state) for state in states]
        next_states = [np.atleast_1d(state) for state in next_states]

        if not states:
            print("Warning: No states were collected during this step.")
            return rewards, dones

        try:
            global_state = np.concatenate(states)
            next_global_state = np.concatenate(next_states)

            self.add_to_buffer((
                states,
                actions,
                rewards,
                next_states,
                dones,
                global_state,
                next_global_state
            ))
        except ValueError as e:
            print(f"Error during concatenation: {e}")
            print(f"States: {states}")
            print(f"Next States: {next_states}")
            raise

        return rewards, dones