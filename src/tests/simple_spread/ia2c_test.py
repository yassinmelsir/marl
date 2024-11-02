from src.agents.a2c.ia2c_agent import Ia2cAgent
from src.agents.common import AgentParams
from src.environment.common import LoopParams
from src.environment.simple_spread import SimpleSpreadParams, SimpleSpread
from src.experiment.training_loop import TrainingLoop



if __name__ == "__main__":
    n_agents=3
    simple_spread_params = SimpleSpreadParams(n=n_agents, local_ratio=0.5, max_cycles=25)
    env_instance = SimpleSpread(params=simple_spread_params)
    loop_params = LoopParams(max_episodes=100, max_timesteps=1000, update_timestep=100)
    agent_params = [
        AgentParams(
            obs_dim = env_instance.obs_size,
            hidden_dim = 128,
            action_dim = env_instance.action_size,
            learning_rate = 0.000001,
            epsilon = 0.2,
            gamma = 0.99,
            K_epochs = 4,
            entropy_coefficient = None
        ) for _ in range(n_agents)]

    agent = Ia2cAgent(agent_params=agent_params)
    train = TrainingLoop(env_instance=env_instance, loop_params=loop_params, agent=agent)
    train.main()