import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from src.transformer.classes.transformer_seq_2_seq import TransformerSeq2Seq


class TrainTransformer:
    def __init__(self, filename, num_heads, num_layers, batch_size, learning_rate, output_filepath):
        loaded_arr = np.load(filename, allow_pickle=True)

        input_sequences = []
        target_sequences = []

        for i in range(len(loaded_arr)):
            sequence = torch.tensor(loaded_arr[i], dtype=torch.float32)
            input_sequences.append(sequence[:-1])
            target_sequences.append(sequence[1:])

        input_sequences = torch.stack(input_sequences)
        target_sequences = torch.stack(target_sequences)

        embed_dim = loaded_arr[0].shape[0]
        train_dataset = TensorDataset(input_sequences, target_sequences)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.model = TransformerSeq2Seq(embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.output_filepath = output_filepath


    def train(self, num_epochs):
        # Training Loop
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            for batch in self.train_loader:
                src, tgt = batch
                self.optimizer.zero_grad()

                # Transpose to match nn.Transformer input (seq_len, batch_size, embed_dim)
                src = src.transpose(0, 1)
                tgt = tgt.transpose(0, 1)

                breakpoint()

                # Forward pass
                output = self.model(src, tgt[:-1])  # Use all but last tgt token as input

                # Calculate loss
                loss = self.criterion(output, tgt[1:])  # Predict next token
                loss.backward()

                # Update weights
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

            torch.save(self.model.state_dict(), f"{self.output_filepath}_{epoch + 1}.pth")
            print(f"Model weights saved for epoch {epoch + 1}")