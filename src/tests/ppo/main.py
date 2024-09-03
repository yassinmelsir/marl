from src.tests.ppo.ppo_test import PpoTest, LoopParams, SimpleSpreadParams, IppoParams

if __name__ == '__main__':
    loop_params = LoopParams(max_episodes=100, max_timesteps=1000, update_timestep=100)
    simple_spread_params = SimpleSpreadParams(n=3, local_ratio=0.5, max_cycles=25)
    ippo_params = IppoParams(lr=3e-4, gamma=0.99, eps_clip=0.2, K_epochs=4)
    ppo_test = PpoTest(loop_params=loop_params, simple_spread_params=simple_spread_params, ippo_params=ippo_params)
    ppo_test.main()