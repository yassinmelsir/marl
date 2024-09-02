import torch
import torch.nn as nn
import torch.nn.functional as F

from value_decomposition.qmix.hyper_network import Hypernetwork


class MixingNetwork(nn.Module):
    def __init__(self, state_dim, n_agents, embed_dim=32):
        super(MixingNetwork, self).__init__()
        self.n_agents = n_agents
        self.hypernetwork = Hypernetwork(state_dim, embed_dim, n_agents)

    def forward(self, agent_qs, state):
        w1, w2, b1, b2 = self.hypernetwork(state)

        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1.unsqueeze(2))
        q_tot = torch.bmm(hidden, w2) + b2.unsqueeze(2)
        q_tot = q_tot.view(-1, 1)

        return q_tot
