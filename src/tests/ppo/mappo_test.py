from src.agents.mappo_agent import MappoAgent
from src.tests.common.common import LoopParams
from src.tests.common.simple_spread import SimpleSpreadParams
from src.tests.ppo.common import PpoParams
from src.tests.ppo.ppo_test import PpoTest

if __name__ == '__main__':
    loop_params = LoopParams(max_episodes=100, max_timesteps=1000, update_timestep=100)
    simple_spread_params = SimpleSpreadParams(n=3, local_ratio=0.5, max_cycles=25)
    ppo_params = PpoParams(agent=MappoAgent, hidden_dim=256, lr=3e-6, gamma=0.99, eps_clip=0.2, K_epochs=4)
    ppo_test = PpoTest(loop_params=loop_params, simple_spread_params=simple_spread_params, ppo_params=ppo_params)
    ppo_test.main()