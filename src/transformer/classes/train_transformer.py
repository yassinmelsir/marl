import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from marl.src.transformer.classes.env_data import EnvData
from marl.src.transformer.classes.transformer_seq_2_seq import TransformerSeq2Seq


class TrainTransformer:
    def __init__(
            self,
            num_heads,
            num_layers,
            batch_size,
            learning_rate,
            output_filepath,
            data=None
    ):
        if data is None:
            data = []
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.output_filepath = output_filepath
        self.data = data

    def load_data_from_file(self, data_filepath):
        self.data = np.load(data_filepath, allow_pickle=True)

    def train(self, num_epochs):
        input_sequences = []
        target_sequences = []

        for i in range(len(self.data)):
            sequence = torch.tensor(self.data[i], dtype=torch.float32)
            input_sequences.append(sequence[:-1])
            target_sequences.append(sequence[1:])

        input_sequences = torch.stack(input_sequences)
        target_sequences = torch.stack(target_sequences)

        embed_dim = input_sequences.shape[1]

        train_dataset = TensorDataset(input_sequences, target_sequences)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        model = TransformerSeq2Seq(embed_dim=embed_dim, num_heads=self.num_heads, num_layers=self.num_layers)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for batch in train_loader:
                src, tgt = batch
                optimizer.zero_grad()

                src = src.transpose(0, 1)
                tgt = tgt.transpose(0, 1)

                output = model(src, tgt[:-1])

                loss = criterion(output, tgt[1:])
                loss.backward()

                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

            if (epoch + 1) % 10 == 0:
                torch.save(model.state_dict(), f"{self.output_filepath}_{epoch + 1}.pth")
                print(f"Model weights saved for epoch {epoch + 1}")
