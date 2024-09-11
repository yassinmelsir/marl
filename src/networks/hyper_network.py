import torch
from torch import nn

class Hypernetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim, n_agents):
        super().__init__()

        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.n_agents = n_agents

        self.w1 = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, n_agents * hidden_dim),
        )

        self.w2 = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.b1 = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
        )
        self.b2 = nn.Sequential(
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, observation):
        w1 = torch.abs(self.w1(observation))
        w1 = w1.view(-1, self.n_agents, self.hidden_dim)

        w2 = self.w2(observation)
        w2 = w2.view(-1, self.hidden_dim, 1)

        b1 = self.b1(observation)
        b2 = self.b2(b1)

        return w1, w2, b1, b2
