import torch
import torch.nn as nn


class ValueCritic(nn.Module):
    def __init__(self, obs_dim, hidden_dim):
        super().__init__()
        self.obs_dim = obs_dim

        self.q_network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs):
        breakpoint()
        return self.q_network(obs)
