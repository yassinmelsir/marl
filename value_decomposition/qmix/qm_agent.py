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

        self.mixing_network = MixingNetwork(
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

        params = list(self.agents.parameters()) + list(self.mixing_network.parameters())
        self.optimizer = torch.optim.Adam(params=params, lr=learning_rate)

        self.replay_buffer = ReplayBuffer(batch_size=batch_size, storage=ListStorage(max_size=buffer_capacity))
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.n_actions = n_actions
        self.q_agent_state_dim = q_agent_state_dim
        self.n_agents = n_agents
        self.update_frequency = update_frequency

    def select_action(self, observation, id, random_possible=True):
        if random_possible and torch.rand(1).item() < self.epsilon:
            action = torch.randint(0, self.n_actions, (1,)).item()
            return action
        else:
            with torch.no_grad():
                action_q_values, _ = self.agents[id](observation.unsqueeze(0))
                action = action_q_values.argmax().item()
                return action

    def update(self, random_possible=False):
        if len(self.replay_buffer) < self.batch_size:
            return None

        q_values_batch, next_q_values_batch, state_batch, next_state_batch, rewards_batch, dones_batch = \
            self.get_batch(random_possible=random_possible)

        joint_q_values = self.mixing_network(q_values_batch[0], state_batch[0])

        with torch.no_grad():
            dones = torch.tensor(dones_batch[0], dtype=torch.float32)
            rewards = torch.tensor(rewards_batch[0], dtype=torch.float32)
            next_q_values = next_q_values_batch[0]

            target_q_values = self.mixing_network(next_q_values[0], next_state_batch[0])
            targets = rewards + (1 - dones) * self.gamma * target_q_values

        loss = F.mse_loss(joint_q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def step(self, env):
        data = {}
        for agent_id in env.agents:
            observation, reward, termination, truncation, info = env.last()

            if np.isscalar(observation):
                observation = np.array(observation)

            if termination or truncation:
                action = None
            else:
                action = self.select_action(observation=torch.FloatTensor(observation), id=agent_id)

            env.step(action)

            next_observation = env.observe(agent_id)[0]
            if np.isscalar(next_observation):
                next_observation = np.array(next_observation)

            data['agent_id'] = {
                'observation': observation, 'reward': reward, 'done': termination or truncation,
                'action': action, 'next_observation': next_observation
            }

        self.add_to_buffer(data)

        rewards, dones = [], []
        for id, agent in data.items():
            rewards.append(agent['reward'])
            dones.append(agent['done'])

        return rewards, dones

    def add_to_buffer(self, data):
        self.replay_buffer.add(data)

    def get_batch(self, random_possible):
        batch = self.replay_buffer.sample(batch_size=self.batch_size)

        q_values_batch = []
        next_q_values_batch = []
        state_batch = []
        next_state_batch = []
        rewards_batch = []
        dones_batch = []

        for step in batch:
            q_values, next_q_values, state, next_state, rewards, dones = [], [], [], [], [], []
            for agent in step.items():
                id = agent['id']
                observation = agent['observation']
                next_observation = agent['next_observation']
                reward = agent['reward']
                done = agent['done']

                action_q = self.select_action(observation=observation, id=id, random_possible=random_possible)
                next_action_q = self.select_action(observation=next_observation, id=id, random_possible=random_possible)

                state.append(observation)
                next_state.append(next_observation)

                q_values.append(action_q)
                next_q_values.append(next_action_q)

                rewards.append(reward)
                dones.append(done)

            q_values_batch.append(q_values)
            next_q_values_batch.append(next_q_values)

            state_batch.append(state)
            next_state_batch.append(next_state)

            rewards_batch.append(rewards)
            dones_batch.append(dones)

        state_batch = [state for state in batch['state']]
        next_state_batch = [state for state in batch['state']]

        return q_values_batch, next_q_values_batch, state_batch, next_state_batch, rewards_batch, dones_batch
