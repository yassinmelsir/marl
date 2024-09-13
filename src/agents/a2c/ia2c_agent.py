import torch

from src.agents.a2c.a2c_agent import A2cAgent
from src.agents.i_agent import IAgent
from src.common.memory import Memory
from src.networks.stochastic_actor import StochasticActor
from src.networks.state_critic import StateCritic


class Ia2cAgent(IAgent):
    def __init__(self, agent_params):
        super().__init__(agent_params, None)
        self.agents = []
        for param in agent_params:
            actor = StochasticActor(obs_dim=param.obs_dim, action_dim=param.action_dim, hidden_dim=param.hidden_dim)
            critic = StateCritic(obs_dim=param.obs_dim, hidden_dim=param.hidden_dim)
            memory = Memory()
            agent = A2cAgent(
                actor=actor,
                critic=critic,
                memory=memory,
                learning_rate=param.learning_rate,
                gamma=param.gamma,
                epsilon=param.epsilon,
                K_epochs=param.K_epochs,
                entropy_coefficient=param.entropy_coefficient
            )
            self.agents.append(agent)
