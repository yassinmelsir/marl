import numpy as np

from src.tests.common.common import LoopParams
from src.tests.common.simple_spread import SimpleSpread, SimpleSpreadParams
from src.tests.a2c.common import A2cParams


class A2cTest:
    def __init__(self, simple_spread_params: SimpleSpreadParams, a2c_params: A2cParams, loop_params: LoopParams):
        self.a2c_params = a2c_params
        self.loop_params = loop_params

        self.simple_spread = SimpleSpread(params=simple_spread_params)
        self.simple_spread.reset()

        obs_dim, action_dim , n_agents = \
            self.simple_spread.obs_size, self.simple_spread.action_size, self.simple_spread.n_agents

        self.a2c_agent = a2c_params.agent(
            n_agents=n_agents,
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=a2c_params.hidden_dim,
            lr=a2c_params.lr,
            gamma=a2c_params.gamma,
            eps_clip=a2c_params.eps_clip,
            K_epochs=a2c_params.K_epochs,
            entropy_coefficient=a2c_params.entropy_coefficient
        )

    def main(self):
        max_episodes = self.loop_params.max_episodes
        max_timesteps = self.loop_params.max_timesteps
        update_timestep = self.loop_params.update_timestep

        timestep = 0
        for episode in range(max_episodes):
            self.simple_spread.reset()
            for t in range(max_timesteps):
                env = self.simple_spread.get_env()
                dones = self.a2c_agent.step(env=env)

                if all(dones):
                    break

                if timestep % update_timestep == 0:
                    self.a2c_agent.update()
                    timestep = 0

                timestep += 1

                if (timestep + 1) % 100 == 0:
                    print(
                        f"timestep {timestep + 1} - average reward: \
                         {np.mean([np.sum(m.rewards) for m in self.a2c_agent.get_memories()])}")

            print(f"Episode {episode + 1} finished")

            if (episode + 1) % 100 == 0:
                print(
                    f"Episode {episode + 1} - average reward: \
                     {np.mean([np.sum(m.rewards) for m in self.a2c_agent.get_memories()])}")
