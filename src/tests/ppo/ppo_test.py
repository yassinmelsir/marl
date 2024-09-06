import numpy as np

from src.tests.common.common import LoopParams
from src.tests.common.simple_spread import SimpleSpread, SimpleSpreadParams
from src.tests.ppo.common import PpoParams


class PpoTest:
    def __init__(self, simple_spread_params: SimpleSpreadParams, ppo_params: PpoParams, loop_params: LoopParams):
        self.ppo_params = ppo_params
        self.loop_params = loop_params

        self.simple_spread = SimpleSpread(params=simple_spread_params)
        self.simple_spread.reset()

        obs_dim, action_dim , n_agents = \
            self.simple_spread.obs_size, self.simple_spread.action_size, self.simple_spread.n_agents

        self.ppo_agent = ppo_params.agent(
            n_agents=n_agents,
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=ppo_params.hidden_dim,
            lr=ppo_params.lr,
            gamma=ppo_params.gamma,
            eps_clip=ppo_params.eps_clip,
            K_epochs=ppo_params.K_epochs
        )

    def main(self):
        max_episodes = self.loop_params.max_episodes
        max_timesteps = self.loop_params.max_timesteps
        update_timestep = self.loop_params.update_timestep

        for episode in range(max_episodes):
            self.simple_spread.reset()
            timestep = 0

            for t in range(max_timesteps):
                env = self.simple_spread.get_env()
                dones = self.ppo_agent.step(env=env)

                if all(dones):
                    break

                if timestep % update_timestep == 0:
                    self.ppo_agent.update()

                timestep += 1

                if (timestep + 1) % 100 == 0:
                    print(
                        f"timestep {timestep + 1} - average reward: \
                         {np.mean([np.sum(m.rewards) for m in self.ppo_agent.get_memories()])}")

            print(f"Episode {episode + 1} finished")

            if (episode + 1) % 100 == 0:
                print(
                    f"Episode {episode + 1} - average reward: \
                     {np.mean([np.sum(m.rewards) for m in self.ppo_agent.get_memories()])}")
