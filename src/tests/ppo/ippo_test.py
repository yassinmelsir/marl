from dataclasses import dataclass
import numpy as np

from src.ppo.agents.ippo_agent import IppoAgent
from src.tests.common.common import LoopParams
from src.tests.common.simple_spread import SimpleSpread, SimpleSpreadParams


@dataclass
class IppoParams:
    lr: float
    gamma: float
    eps_clip: float
    K_epochs: int
    hidden_dim: int


class IppoTest:
    def __init__(self, simple_spread_params: SimpleSpreadParams, ippo_params: IppoParams, loop_params: LoopParams):
        self.ippo_params = ippo_params
        self.loop_params = loop_params

        self.simple_spread = SimpleSpread(params=simple_spread_params)
        self.simple_spread.reset()

        obs_dim, action_dim , n_agents = \
            self.simple_spread.obs_size, self.simple_spread.action_size, self.simple_spread.n_agents

        self.ippo_agent = IppoAgent(
            n_agents=n_agents,
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=ippo_params.hidden_dim,
            lr=ippo_params.lr,
            gamma=ippo_params.gamma,
            eps_clip=ippo_params.eps_clip,
            K_epochs=ippo_params.K_epochs
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
                dones = self.ippo_agent.step(env=env)

                if all(dones):
                    break

                if timestep % update_timestep == 0:
                    self.ippo_agent.update()
                    timestep = 0

                timestep += 1

                if (timestep + 1) % 100 == 0:
                    print(
                        f"timestep {timestep + 1} - average reward: \
                         {np.mean([np.sum(m.rewards) for m in self.ippo_agent.get_memories()])}")

            print(f"Episode {episode + 1} finished")

            if (episode + 1) % 100 == 0:
                print(
                    f"Episode {episode + 1} - average reward: \
                     {np.mean([np.sum(m.rewards) for m in self.ippo_agent.get_memories()])}")
