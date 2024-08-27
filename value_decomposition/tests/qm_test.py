import gymnasium as gym
import torch
from pettingzoo.mpe import simple_spread_v3
from value_decomposition.qmix.qm_agent import QmAgent
import numpy as np


def main(num_episodes, max_steps_per_episode, visualize=False):
    env = simple_spread_v3.env(N=3, local_ratio=0.5, max_cycles=max_steps_per_episode)
    env.reset()

    first_agent = env.agents[0]
    observation_space = env.observation_space(first_agent)
    action_space = env.action_space(first_agent)

    if isinstance(observation_space, gym.spaces.Box):
        input_size = observation_space.shape[0]
    else:
        raise ValueError(f"Unexpected observation space type: {type(observation_space)}")

    if isinstance(action_space, gym.spaces.Discrete):
        output_size = action_space.n
    else:
        raise ValueError(f"Unexpected action space type: {type(action_space)}")

    agent = QmAgent(
        n_agents=len(env.agents),
        embed_dim=128,
        mixing_state_dim=input_size * len(env.agents),
        q_agent_state_dim=input_size,
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
        env.reset()
        total_reward = 0
        step = 0

        while step < max_steps_per_episode:

            rewards, dones = agent.step(env=env)

            loss = agent.update()
            if loss is not None:
                print(f"Episode {episode + 1}, Step {step + 1}, Loss: {loss:.4f}")

            total_reward += sum(rewards)
            step += 1

            if visualize:
                env.render()

            if all(dones):
                break

        print(f"Episode {episode + 1}, Total Reward: {total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main(num_episodes=100, max_steps_per_episode=500)
