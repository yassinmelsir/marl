from src.agents.common import AgentParams, CentralParams
from src.agents.ppo.ippo_agent import IppoAgent
from src.common.replay_buffer import ReplayBuffer
from src.environment.common import LoopParams
from src.environment.simple_spread import SimpleSpreadParams, SimpleSpread
from src.experiment.training_loop import TrainingLoop

if __name__ == "__main__":
    simple_spread_params = SimpleSpreadParams(n=3, local_ratio=0.5, max_cycles=25)
    env_instance = SimpleSpread(params=simple_spread_params)
    loop_params = LoopParams(max_episodes=100, max_timesteps=1000, update_timestep=100)
    n_agents=3
    hidden_dim=128
    learning_rate=0.000001
    epsilon=0.2
    gamma=0.99
    obs_dim=env_instance.obs_size
    replay_buffer=ReplayBuffer(buffer_capacity=1000, batch_size=5)
    action_dim=env_instance.action_size
    K_epochs = 4

    agent_params = [
        AgentParams(
            obs_dim=env_instance.obs_size,
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
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        replay_buffer=replay_buffer
    )
    agent = IppoAgent(agent_params=agent_params)
    vdn_test = TrainingLoop(env_instance=env_instance, loop_params=loop_params, agent=agent)
    vdn_test.main()