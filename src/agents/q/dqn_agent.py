import numpy as np
import torch
from torch import nn


class DqnAgent:
    def __init__(self, q_network, target_q_network, optimizer, replay_buffer, action_dim, epsilon, gamma):
        self.q_network = q_network
        self.target_network = target_q_network
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer
        self.memory = None

        self.action_dim = action_dim
        self.epsilon = epsilon
        self.gamma = gamma

    def select_action(self, observation):
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation)

        observation = observation.reshape(1, -1)

        observation = torch.FloatTensor(observation)

        action_q_values, _ = self.q_network(observation)

        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                action = action_q_values.argmax().item()

        return action, action_q_values

    def max_action_q_value(self, observation):
        with torch.no_grad():
            action_q_values, _ = self.q_network(observation.unsqueeze(0))
            q_value = action_q_values.max()
            return q_value

    def update(self):
        if self.replay_buffer.can_sample():

            observations, next_observations, actions, _, rewards, dones = self.get_batch()

            action_q_values, _ = self.q_network(observations)
            next_action_q_values, _ = self.target_network(next_observations)

            q_value = action_q_values.sum(dim=1, keepdim=True)
            next_q_value = next_action_q_values.sum(dim=1, keepdims=True)
            expected_q_value = rewards + self.gamma * next_q_value * (1 - dones.float())

            loss = nn.MSELoss()(q_value, expected_q_value)

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

    def get_batch(self):
        batch = self.replay_buffer.sample()
        observations, next_observations, actions, action_qs, rewards, dones  = zip(*batch)

        return (
            torch.stack(observations),
            torch.stack(next_observations),
            torch.stack(actions),
            torch.stack(action_qs),
            torch.stack(rewards),
            torch.stack(dones)
        )