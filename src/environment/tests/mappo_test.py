from src.agents.mappo_agent import MappoAgent
from src.environment.common.common import LoopParams
from src.environment.common.simple_spread import SimpleSpreadParams
from src.environment.ppo.common import PpoParams
from src.environment.ppo.ppo_test import PpoTest

from src.environment.simple_spread import SimpleSpread

if __name__ == '__main__':
    loop_params = LoopParams(max_episodes=100, max_timesteps=1000, update_timestep=100)
    simple_spread_params = SimpleSpreadParams(n=3, local_ratio=0.5, max_cycles=25)
    ppo_params = PpoParams(agent=MappoAgent, hidden_dim=256, lr=3e-6, gamma=0.99, eps_clip=0.2, K_epochs=4)
    ppo_test = PpoTest(loop_params=loop_params, simple_spread_params=simple_spread_params, ppo_params=ppo_params)
    ppo_test.main()

if __name__ == "__main__":
    simple_spread_params = SimpleSpreadParams(n=3, local_ratio=0.5, max_cycles=25)
    env_instance = SimpleSpread(params=simple_spread_params)
    loop_params = LoopParams(max_episodes=100, max_timesteps=1000, update_timestep=100)
    agent = MappoAgent(
        n_agents=env_instance.n,
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=ppo_params.hidden_dim,
        lr=ppo_params.lr,
        gamma=ppo_params.gamma,
        eps_clip=ppo_params.eps_clip,
        K_epochs=ppo_params.K_epochs
    )
    train = TrainingEnvironment(env_instance=env_instance, loop_params=loop_params, agent=agent)
    train.main()