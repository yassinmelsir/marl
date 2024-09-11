from src.environment.ddpg.common import loop_params, simple_spread_params, ddpg_params
from src.environment.ddpg.ddpg_test import DdpgTest

if __name__ == '__main__':
    ddpg_test = DdpgTest(loop_params=loop_params, simple_spread_params=simple_spread_params, ddpg_params=ddpg_params)
    ddpg_test.main()