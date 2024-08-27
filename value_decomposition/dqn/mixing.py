import torch.nn as nn
import torch
import torch.nn.functional as F


class MixingNetwork(nn.Module):
    def __init__(self, n_agents, state_dim):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
