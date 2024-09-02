import torch
from torch import nn
import torch.nn.functional as F


class Hypernetwork(nn.Module):
    def __init__(self, state_dim, embed_dim, n_agents):
        super(Hypernetwork, self).__init__()
        self.state_dim = state_dim
        self.embed_dim = embed_dim
        self.n_agents = n_agents

        self.fc1 = nn.Linear(state_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, n_agents * embed_dim)

        self.fc3 = nn.Linear(state_dim, embed_dim)
        self.fc4 = nn.Linear(embed_dim, embed_dim)

        self.fc5 = nn.Linear(state_dim, embed_dim)
        self.fc6 = nn.Linear(embed_dim, 1)

    def forward(self, state):
        w1 = torch.abs(self.fc2(F.elu(self.fc1(state))))
        w1 = w1.view(-1, self.n_agents, self.embed_dim)

        w2 = torch.abs(self.fc4(F.elu(self.fc3(state))))
        w2 = w2.view(-1, self.embed_dim, 1)

        b1 = self.fc5(state)
        b2 = self.fc6(F.elu(b1))

        return w1, w2, b1, b2
