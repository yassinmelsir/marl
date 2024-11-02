import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Load your dictionary-structured dataset
filename = "/Users/yme/Code/York/marl/src/transformer/transfomer_training_data.npy"
loaded_arr = np.load(filename, allow_pickle=True)

# Hyperparameters
embed_dim = 128
num_heads = 4
num_layers = 3
batch_size = 32
num_epochs = 10
learning_rate = 0.001


# Function to process each timestep dictionary into a flat vector
def process_timestep(timestep_data):
    agent_data = []
    for agent, data in timestep_data.items():
        observation = data['observation']
        action = [data['action']]
        reward = [data['reward']]
        # Concatenate observation, action, and reward into one vector
        agent_vector = np.concatenate([observation, action, reward])
        agent_data.append(agent_vector)
    # Flatten data from all agents into a single vector for this timestep
    return np.concatenate(agent_data)


# Prepare data by processing each sequence of timesteps
input_sequences = []
target_sequences = []

for sequence in loaded_arr:
    processed_sequence = []
    for timestep in sequence:
        if len(timestep) == 0:
            breakpoint()
        processed_timestep = process_timestep(timestep)
        processed_sequence.append(processed_timestep)

    processed_sequence = torch.tensor(processed_sequence, dtype=torch.float32)


    input_sequences.append(processed_sequence[:-1])  # All but last timestep
    target_sequences.append(processed_sequence[1:])  # All but first timestep

# Stack the sequences into tensors
input_sequences = torch.stack(input_sequences)
target_sequences = torch.stack(target_sequences)

# Dataloader
train_dataset = TensorDataset(input_sequences, target_sequences)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# Model Definition
class TransformerSeq2Seq(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super(TransformerSeq2Seq, self).__init__()
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers
        )
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, src, tgt):
        output = self.transformer(src, tgt)
        return self.fc_out(output)


# Instantiate Model, Loss, and Optimizer
model = TransformerSeq2Seq(embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        src, tgt = batch
        optimizer.zero_grad()

        # Transpose to match nn.Transformer input (seq_len, batch_size, embed_dim)
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)

        # Forward pass
        output = model(src, tgt[:-1])  # Use all but last tgt token as input

        # Calculate loss
        loss = criterion(output, tgt[1:])  # Predict next token
        loss.backward()

        # Update weights
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
