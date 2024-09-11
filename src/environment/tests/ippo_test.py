from src.agents.ppo.ippo_agent import IppoAgent
from src.environment.common.common import LoopParams
from src.environment.common.simple_spread import SimpleSpreadParams
from src.environment.ppo.common import PpoParams
from src.environment.ppo.ppo_test import PpoTest

if __name__ == '__main__':
    loop_params = LoopParams(max_episodes=100, max_timesteps=1000, update_timestep=100)
    simple_spread_params = SimpleSpreadParams(n=3, local_ratio=0.5, max_cycles=25)
    ppo_params = PpoParams(agent=IppoAgent, hidden_dim=256, lr=3e-6, gamma=0.99, eps_clip=0.2, K_epochs=4)
    ppo_test = PpoTest(loop_params=loop_params, simple_spread_params=simple_spread_params, ppo_params=ppo_params)
    ppo_test.main()