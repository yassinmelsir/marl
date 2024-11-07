import torch

from src.agents.common import AgentParams, CentralParams
from src.agents.ddpg.maddpg_agent import MaddpgAgent
from src.environment.common import LoopParams
from src.environment.simple_spread import SimpleSpreadParams, SimpleSpread
from src.experiment.training_loop import TrainingLoop
from src.transformer.transformer_seq_2_seq import TransformerSeq2Seq

if __name__ == "__main__":
    n_agents = 3
    simple_spread_params = SimpleSpreadParams(n=n_agents, local_ratio=0.5, max_cycles=25)
    env_instance = SimpleSpread(params=simple_spread_params)
    loop_params = LoopParams(max_episodes=100, max_timesteps=1000, update_timestep=100)
    obs_dim = env_instance.obs_size
    hidden_dim = 128
    action_dim = env_instance.action_size
    learning_rate = 0.000001
    epsilon = 0.2
    gamma = 0.99
    buffer_capacity = 10000
    batch_size = 5
    noise_scale = None
    temperature = 1
    K_epochs = 4

    agent_params = [
        AgentParams(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            learning_rate=learning_rate,
            epsilon=epsilon,
            gamma=gamma,
            buffer_capacity=buffer_capacity,
            batch_size=batch_size,
            temperature=temperature,
            K_epochs=K_epochs
        )
        for _ in range(n_agents)
    ]

    seq_dim = 60
    num_heads = 12
    num_layers = 3
    full_length_srcs = True
    model = TransformerSeq2Seq(seq_dim, num_heads, num_layers)

    model.load_state_dict(torch.load('/Users/yme/Code/York/marl/src/transformer/weights/transformer_epoch_30.pth'))

    central_params = CentralParams(
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        gamma=gamma,
        action_dim=action_dim,
        transformer=model,
        batch_size=batch_size,
        full_length_srcs=full_length_srcs
    )

    agent = MaddpgAgent(agent_params=agent_params, central_params=central_params)
    train = TrainingLoop(env_instance=env_instance, loop_params=loop_params, agent=agent)
    train.main()