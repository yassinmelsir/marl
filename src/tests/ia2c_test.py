from src.agents.a2c.ia2c_agent import Ia2cAgent
from src.environment.common import LoopParams
from src.environment.simple_spread import SimpleSpreadParams, SimpleSpread
from src.environment.training_environment import TrainingEnvironment

if __name__ == "__main__":
    simple_spread_params = SimpleSpreadParams(n=3, local_ratio=0.5, max_cycles=25)
    env_instance = SimpleSpread(params=simple_spread_params)
    loop_params = LoopParams(max_episodes=100, max_timesteps=1000, update_timestep=100)
    agent = Ia2cAgent(
        n_agents=env_instance.n_agents,
        obs_dim=env_instance.obs_size,
        hidden_dim=128,
        action_dim=env_instance.action_size,
        learning_rate=0.000001,
        epsilon=0.2,
        gamma=0.99,
        K_epochs=4,
        entropy_coefficient=None,
    )
    train = TrainingEnvironment(env_instance=env_instance, loop_params=loop_params, agent=agent)
    train.main()