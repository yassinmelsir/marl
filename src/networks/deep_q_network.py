import torch.nn as nn


class DeepQNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim, hidden_output_dim, action_dim):
        super().__init__()
        self.mlp_bottom = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU()
        )
        self.gru = nn.GRU(
            hidden_dim,
            hidden_output_dim,
            batch_first=True
        )
        self.mlp_top = nn.Sequential(
            nn.Linear(hidden_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x, hidden=None):
        x = self.mlp_bottom(x)
        x = x.unsqueeze(1)
        x, hidden = self.gru(x, hidden)
        x = x.squeeze(1)
        x = self.mlp_top(x)
        return x, hidden







