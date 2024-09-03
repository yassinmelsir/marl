from pettingzoo.mpe import simple_spread_v3
from src.coop.qmix.qm_agent import QmAgent
from src.tests.common import get_obs_action_size


def main(num_episodes, max_steps_per_episode, visualize=False):
    env = simple_spread_v3.env(N=3, local_ratio=0.5, max_cycles=max_steps_per_episode)
    env.reset()

    obs_size, action_size = get_obs_action_size(env=env)

    agent = QmAgent(
        n_agents=len(env.agents),
        embed_dim=256,
        mixing_state_dim=obs_size * len(env.agents),
        q_agent_state_dim=obs_size,
        hidden_dim=128,
        hidden_output_dim=32,
        n_actions=action_size,
        learning_rate=0.000001,
        epsilon=0.1,
        gamma=0.99,
        buffer_capacity=10000,
        batch_size=20,
    )

    for episode in range(num_episodes):
        env.reset()
        total_reward = 0
        step = 0

        while step < max_steps_per_episode:
            rewards, dones = agent.step(env=env, step=step)

            loss = agent.update()

            total_reward += sum(rewards)
            step += 1

            if loss is not None:
                print(f"Episode {episode + 1}, Step {step + 1}, Total Reward: {total_reward}, Loss: {loss:.4f}")
            else:
                print(f"Episode {episode + 1}, Step {step + 1}, Total Reward: {total_reward}")

            if all(dones) and len(dones) != 0:
                break


        print(f"Episode {episode + 1}, Total Reward: {total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main(num_episodes=20, max_steps_per_episode=20)
