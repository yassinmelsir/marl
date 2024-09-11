import torch
import torch.nn as nn


class ValueCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.q_network = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, observation, actions):
        x = torch.cat([observation, actions], dim=-1)
        q_value = self.q_network(x)
        return q_value