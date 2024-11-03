import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.transformer.transformer_seq_2_seq import TransformerSeq2Seq

# Load your dictionary-structured dataset
filename = "/Users/yme/Code/York/marl/src/transformer/data/transfomer_training_data.npy"
loaded_arr = np.load(filename, allow_pickle=True)


input_sequences = []
target_sequences = []

for i in range(len(loaded_arr)):
    sequence = torch.tensor(loaded_arr[i], dtype=torch.float32)
    input_sequences.append(sequence[:-1])
    target_sequences.append(sequence[1:])

# Stack the sequences into tensors
input_sequences = torch.stack(input_sequences)
target_sequences = torch.stack(target_sequences)

embed_dim = loaded_arr[0].shape[1]
num_heads = 8
num_layers = 3
batch_size = 32
num_epochs = 10
learning_rate = 0.001

# Dataloader
train_dataset = TensorDataset(input_sequences, target_sequences)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# Model Definitio


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

        breakpoint()

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

    torch.save(model.state_dict(), f"weights/transformer_epoch_{epoch + 1}.pth")
    print(f"Model weights saved for epoch {epoch + 1}")


