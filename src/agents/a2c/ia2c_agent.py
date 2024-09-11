import torch

from src.agents.a2c.a2c_agent import A2cAgent
from src.agents.im_agent import ImAgent
from src.common.memory import Memory
from src.networks.stochastic_actor import StochasticActor
from src.networks.state_critic import StateCritic


class Ia2cAgent(ImAgent):
    def __init__(self, n_agents, obs_dim, action_dim, hidden_dim, learning_rate, gamma, epsilon, K_epochs, entropy_coefficient):
        super().__init__(n_agents, obs_dim, action_dim, hidden_dim, learning_rate, gamma, epsilon, K_epochs)
        self.agents = []
        for _ in range(n_agents):
            actor = StochasticActor(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)
            critic = StateCritic(obs_dim=obs_dim, hidden_dim=hidden_dim)
            memory = Memory()
            agent = A2cAgent(
                actor=actor,
                critic=critic,
                memory=memory,
                learning_rate=learning_rate,
                gamma=gamma,
                epsilon=epsilon,
                K_epochs=K_epochs,
                entropy_coefficient=entropy_coefficient
            )
            self.agents.append(agent)
