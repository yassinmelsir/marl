from src.agents.a2c.a2c_agent import A2cAgent
from src.agents.maacc_agent import Maacc
from src.common.memory import Memory
from src.networks.stochastic_actor import StochasticActor


class Maa2cAgent(Maacc):
    def __init__(self, agent_params, central_params):
        super().__init__(agent_params, central_params)
        self.agents = []
        self.memories = []
        for param in agent_params:
            actor = StochasticActor(obs_dim=param.obs_dim, action_dim=param.action_dim, hidden_dim=param.hidden_dim)
            memory = Memory()
            a2c_agent = A2cAgent(
                actor=actor,
                critic=self.centralized_critic,
                memory=memory,
                learning_rate=param.learning_rate,
                gamma=param.gamma,
                epsilon=param.epsilon,
                K_epochs=param.K_epochs,
                entropy_coefficient=param.entropy_coefficient
            )
            self.agents.append(a2c_agent)
