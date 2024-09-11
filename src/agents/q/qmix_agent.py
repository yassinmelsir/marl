import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from src.common.replay_buffer import ReplayBuffer
from src.networks.deep_q_network import DeepQNetwork
from src.networks.mixing_network import MixingNetwork


class QmixAgent:
    def __init__(self, n_agents, embed_dim, mixing_state_dim,
                 q_agent_state_dim, hidden_dim, hidden_output_dim, n_actions,
                 learning_rate, epsilon, gamma, buffer_capacity, batch_size):

        self.mixing_network = MixingNetwork(
            n_agents=n_agents,
            state_dim=mixing_state_dim,
            embed_dim=embed_dim
        )

        self.agents = nn.ModuleList([
            DeepQNetwork(
                state_dim=q_agent_state_dim,
                hidden_dim=hidden_dim,
                hidden_output_dim=hidden_output_dim,
                n_actions=n_actions
            )
            for _ in range(n_agents)])

        params = list(self.agents.parameters()) + list(self.mixing_network.parameters())
        self.optimizer = torch.optim.Adam(params=params, lr=learning_rate)

        self.replay_buffer = ReplayBuffer(batch_size=batch_size, buffer_size=buffer_capacity)
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.n_actions = n_actions
        self.q_agent_state_dim = q_agent_state_dim
        self.n_agents = n_agents

    def select_action(self, observation, id, random_possible=True):
        if random_possible and torch.rand(1).item() < self.epsilon:
            return torch.randint(0, self.n_actions, (1,)).item()
        else:
            with torch.no_grad():
                action_q_values, _ = self.agents[int(id)](observation.unsqueeze(0))
                return action_q_values.argmax().item()

    def max_action_q_value(self, observation, id):
        with torch.no_grad():
            action_q_values, _ = self.agents[int(id)](observation.unsqueeze(0))
            q_value = action_q_values.max()
            return q_value

    def update(self):
        if self.replay_buffer.can_sample():
            observations, next_observations, rewards, dones = self.get_batch()

            q_values_batch, next_q_values_batch = [], []
            for i in range(len(observations)):
                state, next_state = observations[i], next_observations[i]
                q_values, next_q_values = [], []
                for id in range(len(state)):
                    action_q = self.max_action_q_value(observation=state[id], id=id)
                    next_action_q = self.max_action_q_value(observation=next_state[id], id=id)
                    q_values.append(action_q)
                    next_q_values.append(next_action_q)

                q_values_batch.append(torch.tensor(q_values, dtype=torch.float32, requires_grad=True))
                next_q_values_batch.append(torch.tensor(next_q_values, dtype=torch.float32, requires_grad=True))

            q_values_batch = torch.stack(q_values_batch).reshape(self.batch_size, -1)
            next_q_values_batch = torch.stack(next_q_values_batch).reshape(self.batch_size,-1)

            state_batch = observations.reshape(self.batch_size, -1)
            next_state_batch = next_observations.reshape(self.batch_size, -1)

            global_q_value = self.mixing_network(q_values_batch, state_batch)
            next_global_q_value = self.mixing_network(next_q_values_batch, next_state_batch)

            rewards = rewards.sum(dim=1, keepdim=True)
            dones = dones.float().sum(dim=1, keepdim=True)

            with torch.no_grad():
                y_tot = rewards + self.gamma * (1 - dones) * next_global_q_value

            loss = F.mse_loss(y_tot, global_q_value)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            return loss.item()

    def get_batch(self):
        batch = self.replay_buffer.sample()
        observations, next_observations, rewards, dones = zip(*batch)

        return (
            torch.stack(observations),
            torch.stack(next_observations),
            torch.stack(rewards),
            torch.stack(dones)
        )

    def step(self, env):
        states = []
        next_states = []
        rewards = []
        dones = []
        for idx, agent_id in enumerate(env.agents):
            observation, reward, termination, truncation, _ = env.last()
            obs_tensor = torch.FloatTensor(observation)

            if termination or truncation:
                return rewards, [True]
            else:
                action = self.select_action(observation=obs_tensor, id=idx)

            env.step(action)
            next_observation = env.observe(agent_id)

            next_obs_tensor = torch.FloatTensor(next_observation)
            done_tensor = torch.BoolTensor([termination or truncation])
            reward_tensor = torch.FloatTensor([reward])

            states.append(obs_tensor)
            next_states.append(next_obs_tensor)
            rewards.append(reward_tensor)
            dones.append(done_tensor)

        states = torch.stack(states)
        next_states = torch.stack(next_states)
        rewards = torch.tensor(rewards)
        dones = torch.tensor(dones)

        experience = (
            states,
            next_states,
            rewards,
            dones
        )

        self.replay_buffer.add(experience)

        return rewards, dones

