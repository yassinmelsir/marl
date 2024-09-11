import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from src.agents.q.dqn_agent import DqnAgent
from src.common.replay_buffer import ReplayBuffer
from src.networks.deep_q_network import DeepQNetwork

class VdnAgent:
    def __init__(self, n_agents, state_dim, hidden_dim, hidden_output_dim, action_dim,
                 learning_rate, epsilon, gamma, buffer_capacity, batch_size):
        self.optimizer = None
        params = []
        self.agents = []
        for _ in range(n_agents):
            q_network = DeepQNetwork(state_dim, hidden_dim, hidden_output_dim, action_dim)
            target_q_network = DeepQNetwork(state_dim, hidden_dim, hidden_output_dim, action_dim)
            target_q_network.load_state_dict(q_network.state_dict())
            replay_buffer = ReplayBuffer(batch_size=batch_size, buffer_size=buffer_capacity)
            agent = DqnAgent(
                q_network=q_network,
                target_q_network=target_q_network,
                optimizer=self.optimizer,
                replay_buffer=replay_buffer,
                epsilon=epsilon,
                gamma=gamma,
                action_dim=action_dim
            )
            self.agents.append(agent)
            params.append(q_network.parameters())

        self.optimizer = torch.optim.Adam(params=itertools.chain(*params), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(batch_size=batch_size, buffer_size=buffer_capacity)

        self.batch_size = batch_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.action_dim = action_dim
        self.n_agents = n_agents

    def update(self):
        if self.replay_buffer.can_sample():
            observations, next_observations, actions, rewards, dones = self.get_batch()

            q_values_batch, next_q_values_batch = [], []
            for i in range(len(observations)):
                state, next_state = observations[i], next_observations[i]
                q_values, next_q_values = [], []
                for id in range(len(state)):
                    action_q = self.agents[id].max_action_q_value(observation=state[id])
                    next_action_q = self.agents[id].max_action_q_value(observation=next_state[id])
                    q_values.append(action_q)
                    next_q_values.append(next_action_q)

                q_values_batch.append(torch.tensor(q_values, dtype=torch.float32, requires_grad=True))
                next_q_values_batch.append(torch.tensor(next_q_values, dtype=torch.float32, requires_grad=True))

            q_values_batch = torch.stack(q_values_batch).reshape(self.batch_size, -1)
            next_q_values_batch = torch.stack(next_q_values_batch).reshape(self.batch_size,-1)

            global_q_value = q_values_batch.sum(dim=1, keepdim=True)
            next_global_q_value = next_q_values_batch.sum(dim=1, keepdim=True)
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
        observations, next_observations, actions, rewards, dones = zip(*batch)

        return (
            torch.stack(observations),
            torch.stack(next_observations),
            torch.stack(actions),
            torch.stack(rewards),
            torch.stack(dones)
        )


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
        actions = torch.stack(actions)
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
