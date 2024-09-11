import numpy as np
from src.agents.q.qmix_agent import QmixAgent
from src.tests.common.common import LoopParams
from src.tests.common.simple_spread import SimpleSpread, SimpleSpreadParams

class QmixTest:
    def __init__(self, simple_spread_params: SimpleSpreadParams, loop_params: LoopParams):
        self.loop_params = loop_params

        self.simple_spread = SimpleSpread(params=simple_spread_params)
        self.simple_spread.reset()

        obs_dim, action_dim, n_agents = \
            self.simple_spread.obs_size, self.simple_spread.action_size, self.simple_spread.n_agents

        self.agent = QmixAgent(
            n_agents=n_agents,
            embed_dim=256,
            mixing_state_dim=obs_dim * n_agents,
            q_agent_state_dim=obs_dim,
            hidden_dim=128,
            hidden_output_dim=32,
            n_actions=action_dim,
            learning_rate=0.000001,
            epsilon=0.1,
            gamma=0.99,
            buffer_capacity=10000,
            batch_size=20,
        )

    def get_rewards(self, total_reward):
        return np.mean([np.sum(rwds) for rwds in total_reward])

    def main(self):

        timestep = 0
        total_reward = []
        for episode in range(self.loop_params.max_episodes):
            self.simple_spread.reset()
            timestep_reward = []
            for t in range(self.loop_params.max_timesteps):
                env = self.simple_spread.get_env()
                rewards, dones = self.agent.step(env=env)

                if all(dones):
                    break

                if timestep % self.loop_params.update_timestep == 0:
                    self.agent.update()
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


if __name__ == "__main__":
    simple_spread_params = SimpleSpreadParams(n=3, local_ratio=0.5, max_cycles=25)
    loop_params = LoopParams(max_episodes=100, max_timesteps=1000, update_timestep=100)
    vdn_test = QmixTest(simple_spread_params=simple_spread_params, loop_params=loop_params)
    vdn_test.main()