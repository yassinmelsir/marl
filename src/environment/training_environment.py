import numpy as np

from src.environment.common import LoopParams


class TrainingEnvironment:
    def __init__(self, env_instance, agent, loop_params: LoopParams):
        self.loop_params = loop_params
        self.agent = agent
        self.env_instance = env_instance
        self.env_instance.reset()

    def get_rewards(self, total_reward):
        return np.mean([np.sum(rwds) for rwds in total_reward])

    def main(self):
        max_episodes = self.loop_params.max_episodes
        max_timesteps = self.loop_params.max_timesteps
        update_timestep = self.loop_params.update_timestep

        timestep = 0
        total_reward = []
        for episode in range(max_episodes):
            self.env_instance.reset()
            timestep_reward = []
            for t in range(max_timesteps):
                env = self.env_instance.get_env()
                rewards, dones = self.agent.step(env=env)

                if all(dones):
                    break

                if timestep % update_timestep == 0:
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
