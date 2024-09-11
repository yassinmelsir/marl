from src.agents.q.qmix_agent import QmixAgent
from src.environment.common.common import LoopParams
from src.environment.common.simple_spread import SimpleSpread, SimpleSpreadParams
from src.environment.dqn.training_environment import DqnTest

if __name__ == "__main__":
    simple_spread_params = SimpleSpreadParams(n=3, local_ratio=0.5, max_cycles=25)
    env_instance = SimpleSpread(params=simple_spread_params)
    loop_params = LoopParams(max_episodes=100, max_timesteps=1000, update_timestep=100)
    agent = QmixAgent(
        n_agents=env_instance.n_agents,
        mixing_hidden_dim=256,
        mixing_state_dim=env_instance.obs_size * env_instance.n_agents,
        q_agent_state_dim=env_instance.obs_size,
        hidden_dim=128,
        hidden_output_dim=32,
        action_dim=env_instance.action_size,
        learning_rate=0.000001,
        epsilon=0.1,
        gamma=0.99,
        buffer_capacity=10000,
        batch_size=20,
    )
    vdn_test = DqnTest(env_instance=env_instance, loop_params=loop_params, agent=agent)
    vdn_test.main()