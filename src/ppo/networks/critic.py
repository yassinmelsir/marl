import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_dim):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.critic(x)
