import gymnasium as gym

from src.agents.q.dqn_agent import DqnAgent


def main(env_name, num_episodes, max_steps_per_episode, discrete=True, visualize=False):
    env = gym.make(env_name, render_mode="rgb_array")

    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n if discrete else env.action_space.shape[0]

    agent = DqnAgent(
        state_dim=input_size,
        hidden_dim=64,
        hidden_output_dim=32,
        n_actions=output_size,
        learning_rate=0.001,
        epsilon=0.1,
        gamma=0.99,
        buffer_capacity=10000,
        batch_size=64
    )

    for episode in range(num_episodes):
        state, info = env.reset(seed=42)
        total_reward = 0
        if visualize:
            env.render()

        for step in range(max_steps_per_episode):

            next_state, reward, done = agent.step(env=env, state=state)

            agent.update()

            total_reward += reward
            state = next_state

            if done:
                break

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    env.close()


if __name__ == "__main__":
    # Example usage
    main('CartPole-v1', num_episodes=100, max_steps_per_episode=500, visualize=True)
