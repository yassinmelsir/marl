import torch
import torch.nn as nn
import torch.nn.functional as F

class GumbelActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, temperature=1.0):
        super().__init__()
        self.temperature = temperature

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        logits = self.actor(x)
        return logits

    def sample_gumbel_softmax(self, logits):
        gumbel_noise = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(gumbel_noise + 1e-20) + 1e-20)
        y = logits + gumbel_noise
        return F.softmax(y / self.temperature, dim=-1)