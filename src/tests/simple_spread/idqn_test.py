from src.agents.common import AgentParams
from src.agents.q.idqn_agent import IdqnAgent
from src.environment.common import LoopParams
from src.environment.simple_spread import SimpleSpreadParams, SimpleSpread
from src.experiment.training_loop import TrainingLoop

if __name__ == "__main__":
    n_agents = 3
    simple_spread_params = SimpleSpreadParams(n=n_agents, local_ratio=0.5, max_cycles=25)
    env_instance = SimpleSpread(params=simple_spread_params)
    loop_params = LoopParams(max_episodes=100, max_timesteps=1000, update_timestep=100)
    obs_dim = env_instance.obs_size
    hidden_dim = 128
    hidden_output_dim = 32
    action_dim = env_instance.action_size
    learning_rate = 0.000001
    epsilon = 0.1
    gamma = 0.99
    buffer_capacity = 1000
    batch_size = 20
    K_epochs = 4

    agent_params = [AgentParams(
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        hidden_output_dim=hidden_output_dim,
        action_dim=action_dim,
        learning_rate=learning_rate,
        epsilon=epsilon,
        gamma=gamma,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        K_epochs=K_epochs
    ) for _ in range(n_agents)]

    agent = IdqnAgent(agent_params=agent_params)
    vdn_test = TrainingLoop(env_instance=env_instance, loop_params=loop_params, agent=agent)
    vdn_test.main()