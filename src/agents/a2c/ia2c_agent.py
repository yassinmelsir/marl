import torch

from src.agents.a2c.a2c_agent import A2cAgent
from src.common.memory import Memory
from src.networks.stochastic_actor import StochasticActor
from src.networks.state_critic import StateCritic


class Ia2cAgent:
    def __init__(self, n_agents, obs_dim, action_dim, hidden_dim, learning_rate, gamma, epsilon, K_epochs, entropy_coefficient):
        self.a2c_agents = []
        for _ in range(n_agents):
            actor = StochasticActor(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)
            critic = StateCritic(obs_dim=obs_dim, hidden_dim=hidden_dim)
            memory = Memory()
            a2c_agent = A2cAgent(
                actor=actor,
                critic=critic,
                memory=memory,
                learning_rate=learning_rate,
                gamma=gamma,
                epsilon=epsilon,
                K_epochs=K_epochs,
                entropy_coefficient=entropy_coefficient
            )
            self.a2c_agents.append(a2c_agent)

    def step(self, env):
        dones = []
        rewards = []
        for i, agent_id in enumerate(env.agents):
            observation, reward, termination, truncation, _ = env.last()

            if termination or truncation:
                return rewards, [True]
            else:
                action, log_prob = self.a2c_agents[i].select_action(torch.FloatTensor(observation).unsqueeze(0))
                log_prob = log_prob.squeeze()

            env.step(action)

            done = torch.BoolTensor([termination or truncation]).squeeze()
            reward = torch.FloatTensor([reward]).squeeze()
            observation = observation.squeeze()
            next_observation = torch.FloatTensor(env.observe(agent_id))

            self.a2c_agents[i].memory.actions.append(action)
            self.a2c_agents[i].memory.observations.append(observation)
            self.a2c_agents[i].memory.rewards.append(reward)
            self.a2c_agents[i].memory.dones.append(done)
            self.a2c_agents[i].memory.next_observations.append(next_observation)

            dones.append(termination or truncation)
            rewards.append(reward)

        return rewards, dones

    def update(self):
        for idx, agent in enumerate(self.a2c_agents):
            agent.update()
            agent.memory.clear_memory()

    def get_memories(self):
        return [agent.memory for agent in self.a2c_agents]
