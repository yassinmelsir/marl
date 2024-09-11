from typing import Union

import torch

from src.agents.a2c.a2c_agent import A2cAgent
from src.agents.ddpg.ddpg_agent import DdpgAgent
from src.agents.ppo.ppo_agent import PpoAgent
from src.agents.q.dqn_agent import DqnAgent
from src.common.memory import Memory
from src.networks.stochastic_actor import StochasticActor
from src.networks.state_critic import StateCritic


class ImAgent:
    def __init__(self, n_agents, obs_dim, action_dim, hidden_dim, learning_rate, gamma, epsilon, K_epochs):
        self.agents = []
        for _ in range(n_agents):
            actor = StochasticActor(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)
            critic = StateCritic(obs_dim=obs_dim, hidden_dim=hidden_dim)
            memory = Memory()
            agent = PpoAgent(
                actor=actor,
                critic=critic,
                memory=memory,
                learning_rate=learning_rate,
                gamma=gamma,
                epsilon=epsilon,
                K_epochs=K_epochs
            )
            self.agents.append(agent)


    def step(self, env):
        rewards = []
        dones = []

        for i, agent_id in enumerate(env.agents):
            observation, reward, termination, truncation, _ = env.last()
            obs_tensor = torch.FloatTensor(observation)

            if termination or truncation:
                return rewards, [True]
            else:
                action, action_probs = self.agents[i].select_action(obs_tensor)

            env.step(action)

            done = torch.BoolTensor([termination or truncation])
            reward = torch.FloatTensor([reward])
            observation = observation
            next_observation = torch.FloatTensor(env.observe(agent_id))

            self.agents[i].memory.observations.append(observation)
            self.agents[i].memory.next_observations.append(next_observation)
            self.agents[i].memory.actions.append(action)
            self.agents[i].memory.action_probs.append(action_probs)
            self.agents[i].memory.rewards.append(reward)
            self.agents[i].memory.dones.append(done)


            dones.append(done)
            rewards.append(reward)

        return torch.stack(rewards), torch.stack(dones)

    def update(self):
        for idx, agent in enumerate(self.agents):
            agent.update()
            agent.memory.clear_memory()

    def get_memories(self):
        return [agent.memory for agent in self.agents]

