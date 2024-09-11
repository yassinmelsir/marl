import numpy as np
import torch
from torch import nn


class DqnAgent:
    def __init__(self, q_network, target_q_network, optimizer, replay_buffer, action_dim, epsilon, gamma):
        self.q_network = q_network
        self.target_network = target_q_network
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer

        self.action_dim = action_dim
        self.epsilon = epsilon
        self.gamma = gamma

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                if not isinstance(state, np.ndarray):
                    state = np.array(state)

                state = state.reshape(1, -1)

                state = torch.FloatTensor(state)

                action_q_values, _ = self.q_network(state)
                return action_q_values.argmax().item()

    def update(self):
        if self.replay_buffer.can_sample():

            states, next_states, actions, rewards, dones = self.get_batch()

            action_q_values, _ = self.q_network(states)
            next_action_q_values, _ = self.target_network(next_states)

            q_value = action_q_values.sum(dim=1, keepdim=True)
            next_q_value = next_action_q_values.sum(dim=1, keepdims=True)
            expected_q_value = rewards + self.gamma * next_q_value * (1 - dones.float())

            loss = nn.MSELoss()(q_value, expected_q_value)

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

    def get_batch(self):
        batch = self.replay_buffer.sample()
        states, next_states, actions, rewards, dones  = zip(*batch)

        return (
            torch.stack(states),
            torch.stack(next_states),
            torch.stack(actions),
            torch.stack(rewards),
            torch.stack(dones)
        )