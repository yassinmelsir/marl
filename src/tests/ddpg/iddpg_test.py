from src.tests.ddpg.common import loop_params, simple_spread_params, ddpg_params
from src.tests.ddpg.ddpg_test import DdpgTest

if __name__ == '__main__':
    ddpg_test = DdpgTest(loop_params=loop_params, simple_spread_params=simple_spread_params, ddpg_params=ddpg_params)
    ddpg_test.main()