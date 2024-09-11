import numpy as np
import torch
from torch import optim, nn
from src.agents.q.dqn_agent import DqnAgent
from src.common.replay_buffer import ReplayBuffer
from src.networks.deep_q_network import DeepQNetwork


class IdqnAgent:
    def __init__(self, n_agents, state_dim, hidden_dim, hidden_output_dim, action_dim, learning_rate, epsilon, gamma,
                 buffer_capacity, batch_size):
        self.agents = []
        for _ in range(n_agents):
            q_network = DeepQNetwork(state_dim, hidden_dim, hidden_output_dim, action_dim)
            target_q_network = DeepQNetwork(state_dim, hidden_dim, hidden_output_dim, action_dim)
            target_q_network.load_state_dict(q_network.state_dict())
            optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
            replay_buffer = ReplayBuffer(batch_size=batch_size, buffer_size=buffer_capacity)
            agent = DqnAgent(
                q_network=q_network,
                target_q_network=target_q_network,
                optimizer=optimizer,
                replay_buffer=replay_buffer,
                epsilon=epsilon,
                gamma=gamma,
                action_dim=action_dim
            )
            self.agents.append(agent)

        self.replay_buffer = ReplayBuffer(batch_size=batch_size, buffer_size=buffer_capacity)

        self.epsilon = epsilon
        self.gamma = gamma
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size

    def update(self):
        for idx, agent in enumerate(self.agents):
            agent.update()


    def step(self, env):
        states = []
        next_states = []
        rewards = []
        dones = []
        actions = []
        for idx, agent_id in enumerate(env.agents):
            observation, reward, termination, truncation, _ = env.last()
            obs_tensor = torch.FloatTensor(observation)

            if termination or truncation:
                return rewards, [True]
            else:
                action = self.agents[idx].select_action(state=obs_tensor)

            env.step(action)
            next_observation = env.observe(agent_id)

            next_obs_tensor = torch.FloatTensor(next_observation)
            action_tensor = torch.IntTensor([action])
            done_tensor = torch.BoolTensor([termination or truncation])
            reward_tensor = torch.FloatTensor([reward])

            states.append(obs_tensor)
            next_states.append(next_obs_tensor)
            actions.append(action_tensor)
            rewards.append(reward_tensor)
            dones.append(done_tensor)

            experience = (
                obs_tensor,
                next_obs_tensor,
                action_tensor,
                reward_tensor,
                done_tensor
            )

            self.agents[idx].replay_buffer.add(experience)

        states = torch.stack(states)
        next_states = torch.stack(next_states)
        rewards = torch.tensor(rewards)
        dones = torch.tensor(dones)

        experience = (
            states,
            next_states,
            actions,
            rewards,
            dones
        )

        self.replay_buffer.add(experience)

        return rewards, dones