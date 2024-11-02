from src.agents.a2c.maa2c_agent import Maa2cAgent
from src.agents.common import AgentParams, CentralParams
from src.common.replay_buffer import ReplayBuffer
from src.environment.common import LoopParams
from src.environment.simple_spread import SimpleSpreadParams, SimpleSpread
from src.experiment.training_loop import TrainingLoop

if __name__ == "__main__":
    n_agents = 3
    simple_spread_params = SimpleSpreadParams(n=n_agents, local_ratio=0.5, max_cycles=25)
    env_instance = SimpleSpread(params=simple_spread_params)
    loop_params = LoopParams(max_episodes=100, max_timesteps=1000, update_timestep=100)
    n_agents = env_instance.n_agents
    obs_dim = env_instance.obs_size
    hidden_dim = 128
    action_dim = env_instance.action_size
    learning_rate = 0.000001
    epsilon = 0.2
    gamma = 0.99
    K_epochs = 4
    buffer_capacity = 1000
    batch_size = 5

    agent_params = [
        AgentParams(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            learning_rate=learning_rate,
            epsilon=epsilon,
            gamma=gamma,
            K_epochs=K_epochs
        )
        for _ in range(n_agents)
    ]

    central_params = CentralParams(
        obs_dim=obs_dim*n_agents,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        replay_buffer=ReplayBuffer(
            buffer_capacity=buffer_capacity,
            batch_size=batch_size
        )
    )
    agent = Maa2cAgent(agent_params=agent_params, central_params=central_params
    )
    maac_test = TrainingLoop(env_instance=env_instance, loop_params=loop_params, agent=agent)
    maac_test.main()

