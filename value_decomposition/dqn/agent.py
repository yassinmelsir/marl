import numpy as np
import torch
from torch import optim, nn
from torchrl.data import ReplayBuffer, ListStorage

from value_decomposition.dqn.dqn import DqnNetwork


class DqnAgent:
    def __init__(self,
                 input_size, gru_input_size, gru_output_size,
                 output_size, learning_rate, epsilon, gamma,
                 buffer_capacity, batch_size
                 ):
        self.q_network = DqnNetwork(input_size, gru_input_size, gru_output_size, output_size)
        self.target_network = DqnNetwork(input_size, gru_input_size, gru_output_size, output_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(batch_size=buffer_capacity, storage=ListStorage(max_size=buffer_capacity))
        self.epsilon = epsilon
        self.gamma = gamma
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.output_size)
        else:
            with torch.no_grad():
                if not isinstance(state, np.ndarray):
                    state = np.array(state)

                if state.ndim == 1:
                    state = state.reshape(1, -1)
                elif state.ndim > 2:
                    raise ValueError(f"State has too many dimensions: {state.shape}")

                if state.shape[1] != self.input_size:
                    raise ValueError(f"State has {state.shape[1]} features, expected {self.input_size}")

                state = torch.FloatTensor(state)

                q_values, _ = self.q_network(state)
                return q_values.argmax().item()

    def update(self, qmix=False):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.get_batch()

        q_values, _ = self.q_network(states)
        next_q_values, _ = self.target_network(next_states)

        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = rewards + self.gamma * next_q_value * (1 - dones)

        if qmix:
            loss = None
        else:
            loss = nn.MSELoss()(q_value, expected_q_value.detach())

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

    def get_batch(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        states = np.array(states)
        states = torch.FloatTensor(states)

        actions = np.array(actions)
        actions = torch.LongTensor(actions)

        rewards = np.array(rewards)
        rewards = torch.FloatTensor(rewards)

        next_states = np.array(next_states)
        next_states = torch.FloatTensor(next_states)

        dones = np.array(dones)
        dones = torch.FloatTensor(dones)

        return states, actions, rewards, next_states, dones

    def add_to_buffer(self, data):
        self.replay_buffer.add(data)
