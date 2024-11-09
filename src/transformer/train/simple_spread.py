from pettingzoo.mpe import simple_spread_v3
from src.transformer.classes.train_transformer import TrainTransformer

if __name__ == '__main__':
    gather_data = True
    data_filepath = "/src/transformer/data/transfomer_training_data.npy"
    output_filepath = '/Users/yme/Code/York/marl/src/transformer/weights/transfomer_'
    num_heads = 12
    num_layers = 3
    batch_size = 32
    num_epochs = 30
    learning_rate = 0.001
    num_agents = 3
    num_runs = 100
    steps_per_run = 50
    env = simple_spread_v3.parallel_env(N=num_agents)

    train = TrainTransformer(
        data_filepath=data_filepath,
        num_heads=num_heads,
        batch_size=batch_size,
        num_layers=num_layers,
        learning_rate=learning_rate,
        output_filepath=output_filepath,
        num_agents=num_agents,
        num_runs=num_runs,
        steps_per_run=steps_per_run,
        env=env,
        gather_data=gather_data
    )

    train.train(num_epochs=num_epochs)
