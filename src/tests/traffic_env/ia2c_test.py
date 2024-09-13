from src.agents.a2c.ia2c_agent import Ia2cAgent
from src.agents.common import AgentParams
from src.environment.common import LoopParams
from src.environment.traffic_environment import TrafficEnvironment, TrafficEnvironmentParams
from src.experiment.training_loop import TrainingLoop

if __name__ == "__main__":
    sumo_cmd = ""
    env_params = TrafficEnvironmentParams(sumo_cmd=sumo_cmd, gui=False)
    env_instance = TrafficEnvironment(params=env_params)
    agent_params = [AgentParams(
        obs_dim=agent_param[0],
        action_dim=agent_param[1],
        hidden_dim=128,
        learning_rate=0.000001,
        epsilon=0.2,
        gamma=0.99,
        K_epochs=4,
        entropy_coefficient=None,
        replay_buffer=None
    ) for agent_param in env_instance.get_agent_params()]
    loop_params = LoopParams(max_episodes=100, max_timesteps=1000, update_timestep=100)
    agent = Ia2cAgent(agent_params=agent_params)
    train = TrainingLoop(env_instance=env_instance, loop_params=loop_params, agent=agent)
    train.main()
