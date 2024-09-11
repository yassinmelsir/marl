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

    def step(self, env):
        observations = []
        next_observations = []
        rewards = []
        dones = []
        actions = []
        action_probs = []

        for idx, agent_id in enumerate(env.agents):
            observation, reward, termination, truncation, _ = env.last()
            obs_tensor = torch.FloatTensor(observation)

            if termination or truncation:
                return rewards, [True]
            else:
                action_tuple = self.agents[idx].select_action(observation=obs_tensor)
                action, action_probs_tensor = action_tuple

            env.step(action)
            next_observation = env.observe(agent_id)

            next_obs_tensor = torch.FloatTensor(next_observation)
            action_tensor = torch.IntTensor([action])
            done_tensor = torch.BoolTensor([termination or truncation])
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

            if self.agents[idx].memory:
                self.agents[idx].memory.observations.append(obs_tensor)
                self.agents[idx].memory.next_observations.append(next_obs_tensor)
                self.agents[idx].memory.actions.append(action_tensor)
                self.agents[idx].memory.action_probs.append(action_probs_tensor)
                self.agents[idx].memory.rewards.append(reward_tensor)
                self.agents[idx].memory.dones.append(done_tensor)
            elif self.agents[idx].replay_buffer:
                self.agents[idx].replay_buffer.add(experience)
            else:
                raise "Error! No memory or replay buffer!"

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

