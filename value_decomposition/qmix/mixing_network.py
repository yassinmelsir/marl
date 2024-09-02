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
        # print(f"agent_qs_Shape: {agent_qs.shape}. State Shape: {state.shape}")

        # Generate weights and biases from the hypernetwork
        w1, w2, b1, b2 = self.hypernetwork(state)

        # print(f"w1 shape: {w1.shape}, w2 shape: {w2.shape}")
        # print(f"b1 shape: {b1.shape}, b2 shape: {b2.shape}")

        # Reshape agent_qs to [batch_size, 1, n_agents] (should be [1, 1, 3])
        agent_qs = agent_qs.view(-1, 1, self.n_agents)  # Expected shape: [1, 1, 3]

        # Perform the first matrix multiplication: [1, 1, 3] x [1, 3, 128] -> [1, 1, 128]
        hidden = torch.bmm(agent_qs, w1)  # Shape should be [1, 1, 128]

        # print(f"hidden shape: {hidden.shape}")

        # Correct bias addition: Ensure b1 has the shape [1, 1, 128] before addition
        b1 = b1.view(-1, 1, self.embed_dim)  # Reshape b1 to [1, 1, 128] to match hidden
        hidden = F.elu(hidden + b1)  # Shape after addition should be [1, 1, 128]

        # print(f"hidden shape after bias and activation: {hidden.shape}")

        # Perform the second matrix multiplication: [1, 1, 128] x [1, 128, 1] -> [1, 1, 1]
        q_tot = torch.bmm(hidden, w2) + b2.unsqueeze(2)  # Expected shape: [1, 1, 1]

        # Flatten the output to [batch_size, 1]
        q_tot = q_tot.view(-1, 1)

        # print(f"q_tot: {q_tot.shape}")

        return q_tot
