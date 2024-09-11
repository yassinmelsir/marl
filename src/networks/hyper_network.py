import torch
from torch import nn

class Hypernetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, n_agents):
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.n_agents = n_agents

        self.w1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, n_agents * hidden_dim),
        )

        self.w2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.b1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
        )
        self.b2 = nn.Sequential(
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        w1 = torch.abs(self.w1(state))
        w1 = w1.view(-1, self.n_agents, self.hidden_dim)

        w2 = self.w2(state)
        w2 = w2.view(-1, self.hidden_dim, 1)

        b1 = self.b1(state)
        b2 = self.b2(b1)

        return w1, w2, b1, b2
