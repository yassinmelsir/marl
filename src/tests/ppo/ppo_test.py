from dataclasses import dataclass

from pettingzoo.mpe import simple_spread_v3
import numpy as np

from src.indep.ppo.ippo_agent import IppoAgent
from src.tests.common import get_obs_action_size

@dataclass

class IppoParams:
    lr: float
    gamma: float
    eps_clip: float
    K_epochs: int

@dataclass
class SimpleSpreadParams:
    n: int
    local_ratio: float
    max_cycles: int

@dataclass
class LoopParams:
    max_episodes: int
    max_timesteps: int
    update_timestep: int


class PpoTest:
    def __init__(self, simple_spread_params: SimpleSpreadParams, ippo_params: IppoParams, loop_params: LoopParams):
        self.simple_spread_params = simple_spread_params
        self.ippo_params = ippo_params
        self.loop_params = loop_params

        self.env = simple_spread_v3.env(
            N=simple_spread_params.n,
            local_ratio=simple_spread_params.local_ratio,
            max_cycles=simple_spread_params.max_cycles
        )

        self.env.reset()


        obs_dim, action_dim = get_obs_action_size(env=self.env)

        self.ippo_agent = IppoAgent(
            n_agents=len(self.env.agents),
            obs_dim=obs_dim,
            action_dim=action_dim,
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
            self.env.reset()
            for t in range(max_timesteps):

                dones = self.ippo_agent.step(env=self.env)

                if all(dones):
                    break

                if timestep % update_timestep == 0:
                    self.ippo_agent.update()
                    timestep = 0

                timestep += 1

                if (timestep + 1) % 100 == 0:
                    print(
                        f"timestep {timestep + 1} - average reward: {np.mean([np.sum(m.rewards) for m in self.ippo_agent.get_memories()])}")

            print(f"Episode {episode + 1} finished")

            if (episode + 1) % 100 == 0:
                print(
                    f"Episode {episode + 1} - average reward: {np.mean([np.sum(m.rewards) for m in self.ippo_agent.get_memories()])}")