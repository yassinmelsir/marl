import gymnasium as gym
import numpy as np
import torch

from value_decomposition.dqn.agent import DqnAgent


def main(env_name, num_episodes, max_steps_per_episode, discrete=True):
    env = gym.make(env_name)

    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n if discrete else env.action_space.shape[0]

    agent = DqnAgent(
        input_size=input_size,
        gru_input_size=64,
        gru_output_size=32,
        output_size=output_size,
        learning_rate=0.001,
        epsilon=0.1,
        gamma=0.99,
        buffer_capacity=10000,
        batch_size=64
    )

    for episode in range(num_episodes):
        state, info = env.reset(seed=42)
        total_reward = 0

        for step in range(max_steps_per_episode):

            action = agent.select_action(state)

            next_state, reward, done, _, _ = env.step(action)

            agent.add_to_buffer((state, action, reward, next_state, done))

            agent.update()

            total_reward += reward
            state = next_state

            if done:
                break

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    env.close()


if __name__ == "__main__":
    # Example usage
    main('CartPole-v1', num_episodes=100, max_steps_per_episode=500)
