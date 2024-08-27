import torch.nn as nn


class DqnNetwork(nn.Module):
    def __init__(self, input_size, gru_input_size, gru_output_size, output_size):
        super().__init__()
        self.mlp_bottom = nn.Sequential(
            nn.Linear(input_size, gru_input_size),
            nn.ReLU()
        )
        self.gru = nn.GRU(
            gru_input_size,
            gru_output_size,
            batch_first=True
        )
        self.mlp_top = nn.Sequential(
            nn.Linear(gru_output_size, gru_input_size),
            nn.ReLU(),
            nn.Linear(gru_input_size, output_size)
        )

    def forward(self, x, hidden=None):
        x = self.mlp_bottom(x)
        x = x.unsqueeze(1)
        x, hidden = self.gru(x, hidden)
        x = x.squeeze(1)
        x = self.mlp_top(x)
        return x, hidden








