import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchrl.data import ReplayBuffer, ListStorage
from _collections import defaultdict

from value_decomposition.dqn.deep_q_network import DeepQNetwork
from value_decomposition.qmix.mixing_network import MixingNetwork


class QmAgent:
    def __init__(self, n_agents, embed_dim, mixing_state_dim,
                 q_agent_state_dim, hidden_dim, hidden_output_dim, n_actions,
                 learning_rate, epsilon, gamma, buffer_capacity, batch_size, update_frequency):

        self.q_mixing_network = MixingNetwork(
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

        self.replay_buffer = ReplayBuffer(batch_size=batch_size, storage=ListStorage(max_size=buffer_capacity))
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.n_actions = n_actions
        self.q_agent_state_dim = q_agent_state_dim
        self.n_agents = n_agents
        self.update_frequency = update_frequency

    def select_action(self, state, id, random_possible=True):
        if random_possible and torch.rand(1).item() < self.epsilon:
            action = torch.randint(0, self.n_actions, (1,)).item()
            return action
        else:
            with torch.no_grad():
                action_q_values, _ = self.agents[id](state.unsqueeze(0))
                action = action_q_values.argmax().item()
                return action

    def update(self, random_possible=False):
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(batch_size=self.batch_size)

        q_values_batch = []
        for step in batch:
            agent_action_q_values = []
            for agent in step['agents']:
                state = agent['state']
                id = agent['id']
                selected_action_q_value = self.select_action(state=state, id=id, random_possible=random_possible)
                agent_action_q_values.append(selected_action_q_value)
            q_values_batch.append(agent_action_q_values)

        state_batch = [state for state in batch['state']]

        joint_q_values = self.q_mixing_network(q_values_batch[0], state_batch[0])



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

    def add_to_buffer(self, data):
        self.replay_buffer.add(data)

    def step(self, env):
        step_data = {'agent': [], 'state': [], 'next_state': []}
        for agent_id in env.agents:
            observation, reward, termination, truncation, info = env.last()

            if np.isscalar(observation):
                observation = np.array(observation)

            if termination or truncation:
                action = None
            else:
                action = self.select_action(state=torch.FloatTensor(observation), id=agent_id)

            env.step(action)

            next_observation = env.observe(agent_id)[0]
            if np.isscalar(next_observation):
                next_observation = np.array(next_observation)

            step_data['agents'].append({
                'id': agent_id, 'observation': observation, 'reward': reward, 'done': termination or truncation,
                'action': action, 'next_observation': next_observation
            })

        self.add_to_buffer(step_data)

        rewards, dones = [],[]
        for agent in step_data['agents'].items():
            rewards.append(agent['reward'])
            dones.append(agent['dones'])

        return rewards, dones