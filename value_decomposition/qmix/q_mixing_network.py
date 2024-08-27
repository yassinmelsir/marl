import torch.nn as nn
import torch
import torch.nn.functional as F


class QMixingNetwork(nn.Module):
    def __init__(self, n_agents, state_dim, embed_dim):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        self.hyper_w_one = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
        self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)

        self.hyper_b = nn.Linear(self.state_dim, self.embed_dim)

        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )

    def forward(self, q_values, states):
        batch_size = q_values.size(0)
        states = states.reshape(-1, self.state_dim)
        q_values = q_values.view(-1, 1, self.n_agents)

        w_one = torch.abs(self.hyper_w_one(states))
        w_one = w_one.view(-1, self.n_agents, self.embed_dim)

        b_one = self.hyper_b(states)
        b_one = b_one.view(-1, 1, self.embed_dim)

        hidden = F.elu(torch.bmm(q_values, w_one) + b_one)

        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)

        v = self.V(states).view(-1, 1, 1)

        y = torch.bmm(hidden, w_final) + v

        q_tot = y.view(batch_size, -1, 1)
        return q_tot
