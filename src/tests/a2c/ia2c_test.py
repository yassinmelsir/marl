from src.agents.a2c.ia2c_agent import Ia2cAgent
from src.tests.common.common import LoopParams
from src.tests.common.simple_spread import SimpleSpreadParams
from src.tests.a2c.common import A2cParams
from src.tests.a2c.a2c_test import A2cTest

if __name__ == '__main__':
    loop_params = LoopParams(max_episodes=100, max_timesteps=1000, update_timestep=100)
    simple_spread_params = SimpleSpreadParams(n=3, local_ratio=0.5, max_cycles=25)
    a2c_params = A2cParams(agent=Ia2cAgent, hidden_dim=256, lr=3e-6, gamma=0.99, eps_clip=0.2, K_epochs=4, entropy_coefficient=None)
    a2c_test = A2cTest(loop_params=loop_params, simple_spread_params=simple_spread_params, a2c_params=a2c_params)
    a2c_test.main()