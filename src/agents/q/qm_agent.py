import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchrl.data import ReplayBuffer, ListStorage

from src.networks.deep_q_network import DeepQNetwork
from src.networks.mixing_network import MixingNetwork


class QmAgent:
    def __init__(self, n_agents, embed_dim, mixing_state_dim,
                 q_agent_state_dim, hidden_dim, hidden_output_dim, n_actions,
                 learning_rate, epsilon, gamma, buffer_capacity, batch_size):

        self.mixing_network = MixingNetwork(
            n_agents=n_agents,
            state_dim=mixing_state_dim,
            embed_dim=embed_dim
        )

        self.agents = nn.ModuleList([
            DeepQNetwork(
                state_dim=q_agent_state_dim,
                hidden_dim=hidden_dim,
                hidden_output_dim=hidden_output_dim,
                n_actions=n_actions
            )
            for _ in range(n_agents)])

        params = list(self.agents.parameters()) + list(self.mixing_network.parameters())
        self.optimizer = torch.optim.Adam(params=params, lr=learning_rate)

        self.replay_buffer = ReplayBuffer(batch_size=batch_size, storage=ListStorage(max_size=buffer_capacity))
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.n_actions = n_actions
        self.q_agent_state_dim = q_agent_state_dim
        self.n_agents = n_agents

    def select_action(self, observation, id, random_possible=True):
        if random_possible and torch.rand(1).item() < self.epsilon:
            return torch.randint(0, self.n_actions, (1,)).item()
        else:
            with torch.no_grad():
                action_q_values, _ = self.agents[int(id)](observation.unsqueeze(0))
                return action_q_values.argmax().item()

    def max_action_q_value(self, observation, id):
        with torch.no_grad():
            action_q_values, _ = self.agents[int(id)](observation.unsqueeze(0))
            q_value = action_q_values.max()
            return q_value

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        state_batch, next_state_batch, rewards_batch, dones_batch, step_no = \
            self.replay_buffer.sample()

        for item in [next_state_batch, rewards_batch, dones_batch, step_no]:
            if len(item) != len(state_batch): raise "Length mismatch in batch!"

        q_values_batch, next_q_values_batch = [], []
        for j in range(len(state_batch)):
            state, next_state = state_batch[j], next_state_batch[j]
            q_values, next_q_values = [], []
            for id in range(len(state)):
                action_q = self.max_action_q_value(observation=state[id], id=id)
                next_action_q = self.max_action_q_value(observation=next_state[id], id=id)
                q_values.append(action_q)
                next_q_values.append(next_action_q)

            q_values_batch.append(torch.tensor(np.array(q_values, dtype=np.float32)))
            next_q_values_batch.append(torch.tensor(np.array(q_values, dtype=np.float32)))

        q_values_batch = torch.stack(q_values_batch).reshape(self.batch_size,-1)
        next_q_values_batch = torch.stack(next_q_values_batch).reshape(self.batch_size,-1)

        state_batch = state_batch.reshape(self.batch_size,-1)
        next_state_batch = next_state_batch.reshape(self.batch_size,-1)

        global_q_value = self.mixing_network(q_values_batch, state_batch)
        next_global_q_value = self.mixing_network(next_q_values_batch, next_state_batch)

        if global_q_value.shape != next_global_q_value.shape:
            raise "joint_q_values.shape != target_q_values.shape"

        with torch.no_grad():
            rewards_sum_batch = rewards_batch.sum(dim=1, keepdim=True)
            dones_sum_batch = dones_batch.sum(dim=1, keepdim=True)
            y_tot = rewards_sum_batch + self.gamma * (1 - dones_sum_batch) * next_global_q_value

        loss = F.mse_loss(y_tot, global_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def step(self, env, step):
        state = []
        next_state = []
        rewards = []
        dones = []
        for idx, agent_id in enumerate(env.agents):
            observation, reward, termination, truncation, info = env.last()

            if np.isscalar(observation):
                observation = np.array(observation)

            if termination or truncation:
                action = None
            else:
                action = self.select_action(observation=torch.FloatTensor(observation), id=idx)

            env.step(action)

            next_observation = env.observe(agent_id)
            state.append(np.array(observation, dtype=np.float32))
            next_state.append(np.array(next_observation, dtype=np.float32))
            rewards.append(np.array(reward, dtype=np.float32))
            dones.append(np.array(termination or truncation, dtype=np.float32))

        for item in [state, next_state, rewards, dones]:
            if len(item) != len(state): raise "Item mismatch in step data!"

        state = torch.tensor(np.array(state), dtype=torch.float32)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        dones = torch.tensor(np.array(dones), dtype=torch.float32)

        data = (state, next_state, rewards, dones, step)

        self.replay_buffer.add(data)

        return rewards, dones
