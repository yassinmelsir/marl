import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pettingzoo.mpe import simple_spread_v3
from torch.utils.data import DataLoader, TensorDataset

from src.transformer.classes.gather_transformer_data import GatherTransformerData
from src.transformer.classes.train_transformer import TrainTransformer
from src.transformer.classes.transformer_seq_2_seq import TransformerSeq2Seq

if __name__ == '__main__':
    file_name = "data/transfomer_training_data.npy"

    num_agents = 3
    num_runs = 100
    steps_per_run = 50
    env = simple_spread_v3.parallel_env(N=num_agents)

    train = GatherTransformerData(
        num_agents=num_agents,
        num_runs=num_runs,
        steps_per_run=steps_per_run,
        env=env,
        file_name=file_name
    )

    train.gather_data()
    train.shape_data()
    train.print_data()
    train.save_data()

    filename = "/Users/yme/Code/York/marl/src/transformer/data/transfomer_training_data.npy"
    num_heads = 12
    num_layers = 3
    batch_size = 32
    num_epochs = 30
    learning_rate = 0.001
    output_filepath='weights/transformer'

    train = TrainTransformer(
        filename=filename,
        num_heads=num_heads,
        batch_size=batch_size,
        num_layers=num_layers,
        learning_rate=learning_rate,
        output_filepath=output_filepath
    )

    train.train(num_epochs=num_epochs)





