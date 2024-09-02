import torch
import torch.nn as nn
import torch.nn.functional as F

from value_decomposition.qmix.hyper_network import Hypernetwork


class MixingNetwork(nn.Module):
    def __init__(self, state_dim, n_agents, embed_dim=32):
        super(MixingNetwork, self).__init__()
        self.n_agents = n_agents
        self.hypernetwork = Hypernetwork(state_dim=state_dim, embed_dim=embed_dim, n_agents=n_agents)

        self.embed_dim = embed_dim

    def forward(self, agent_qs, state):
        w1, w2, b1, b2 = self.hypernetwork(state)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        hidden = torch.bmm(agent_qs, w1)

        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(hidden + b1)
        q_tot = torch.bmm(hidden, w2) + b2.unsqueeze(2)
        q_tot = q_tot.view(-1, 1)

        return q_tot
