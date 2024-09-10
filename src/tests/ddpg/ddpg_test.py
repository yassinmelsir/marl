import numpy as np
import torch

from src.tests.common.common import LoopParams
from src.tests.common.simple_spread import SimpleSpread, SimpleSpreadParams
from src.tests.ddpg.common import DdpgParams


class DdpgTest:
    def __init__(self, simple_spread_params: SimpleSpreadParams, ddpg_params: DdpgParams, loop_params: LoopParams):
        self.ddpg_params = ddpg_params
        self.loop_params = loop_params

        self.simple_spread = SimpleSpread(params=simple_spread_params)
        self.simple_spread.reset()

        obs_dim, action_dim , n_agents = \
            self.simple_spread.obs_size, self.simple_spread.action_size, self.simple_spread.n_agents

        self.ddpg_agent = ddpg_params.agent(
            n_agents=n_agents,
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=ddpg_params.hidden_dim,
            lr=ddpg_params.lr,
            gamma=ddpg_params.gamma,
            eps_clip=ddpg_params.eps_clip,
            K_epochs=ddpg_params.K_epochs,
            buffer_size=ddpg_params.buffer_size,
            batch_size=ddpg_params.batch_size,
            noise_scale=ddpg_params.noise_scale,
            temperature=ddpg_params.temperature
        )

    def get_rewards(self, total_reward):
        return np.mean([np.sum(rwds) for rwds in total_reward])

    def main(self):
        max_episodes = self.loop_params.max_episodes
        max_timesteps = self.loop_params.max_timesteps
        update_timestep = self.loop_params.update_timestep

        timestep = 0
        total_reward = []
        for episode in range(max_episodes):
            self.simple_spread.reset()
            timestep_reward = []
            for t in range(max_timesteps):
                env = self.simple_spread.get_env()
                rewards, dones = self.ddpg_agent.step(env=env)

                if all(dones):
                    break

                if timestep % update_timestep == 0:
                    self.ddpg_agent.update()
                    timestep = 0

                timestep_reward.append(np.array(rewards))
                timestep += 1

                if (timestep + 1) % 100 == 0:
                    print(
                        f"timestep {timestep + 1} - average reward: \
                         {self.get_rewards(total_reward=total_reward)}")


            total_reward.append(np.array(timestep_reward))

            print(f"Episode {episode + 1} finished")

            if (episode + 1) % 100 == 0:
                print(
                    f"Episode {episode + 1} - average reward: \
                     {self.get_rewards(total_reward=total_reward)}")
