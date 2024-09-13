from typing import Union

import torch

from src.agents.a2c.a2c_agent import A2cAgent
from src.agents.ddpg.ddpg_agent import DdpgAgent
from src.agents.ppo.ppo_agent import PpoAgent
from src.agents.q.dqn_agent import DqnAgent
from src.common.memory import Memory
from src.common.replay_buffer import ReplayBuffer
from src.networks.stochastic_actor import StochasticActor
from src.networks.state_critic import StateCritic


class IAgent:
    def __init__(self, n_agents, obs_dim, action_dim, hidden_dim, learning_rate, gamma, epsilon, K_epochs, batch_size, buffer_capacity):
        self.agents = []
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.epsilon = epsilon
        self.gamma = gamma
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        if batch_size is not None and buffer_capacity is not None:
            self.replay_buffer = ReplayBuffer(batch_size=batch_size, buffer_capacity=buffer_capacity)
        else:
            self.replay_buffer = None

    def update(self):
        for idx, agent in enumerate(self.agents):
            agent.update()
            agent.memory.clear_memory()

    def get_batch(self):
        batch = self.replay_buffer.sample()
        observations, next_observations, actions, action_probs, rewards, dones = zip(*batch)

        return (
            torch.stack(observations),
            torch.stack(next_observations),
            torch.stack(actions),
            torch.stack(action_probs),
            torch.stack(rewards),
            torch.stack(dones)
        )

    def get_memories(self):
        return [agent.memory for agent in self.agents]

    def save_agent_data(self, global_experience, agent_experience, agent):
        obs_tensor, next_observation, action, action_probs_tensor, reward, done = agent_experience
        observations, next_observations, actions, action_probs, rewards, dones = global_experience

        next_obs_tensor = torch.FloatTensor(next_observation)
        action_tensor = torch.IntTensor([action])
        done_tensor = torch.BoolTensor([done])
        reward_tensor = torch.FloatTensor([reward])

        observations.append(obs_tensor)
        next_observations.append(next_obs_tensor)
        actions.append(action_tensor)
        action_probs.append(action_probs_tensor)
        rewards.append(reward_tensor)
        dones.append(done_tensor)

        experience = (
            obs_tensor,
            next_obs_tensor,
            action_tensor,
            action_probs_tensor,
            reward_tensor,
            done_tensor
        )

        if agent.memory:
            agent.memory.observations.append(obs_tensor)
            agent.memory.next_observations.append(next_obs_tensor)
            agent.memory.actions.append(action_tensor)
            agent.memory.action_probs.append(action_probs_tensor)
            agent.memory.rewards.append(reward_tensor)
            agent.memory.dones.append(done_tensor)
        elif agent.replay_buffer:
            agent.replay_buffer.add(experience)
        else:
            raise "Error! No memory or replay buffer!"

    def save_global_data(self, experience):
        observations, next_observations, actions, action_probs, rewards, dones = experience

        observations = torch.stack(observations)
        next_observations = torch.stack(next_observations)
        actions = torch.stack(actions)
        action_probs = torch.stack(action_probs)
        rewards = torch.tensor(rewards)
        dones = torch.tensor(dones)

        experience = (
            observations,
            next_observations,
            actions,
            action_probs,
            rewards,
            dones
        )

        if self.replay_buffer is not None:
            self.replay_buffer.add(experience)

        return rewards, dones

