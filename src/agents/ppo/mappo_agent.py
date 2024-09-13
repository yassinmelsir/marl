from src.agents.maacc_agent import Maacc
from src.agents.ppo.ppo_agent import PpoAgent
from src.common.memory import Memory
from src.networks.stochastic_actor import StochasticActor


class MappoAgent(Maacc):
    def __init__(self, agent_params, central_params):
        super().__init__(agent_params, central_params)
        self.ppo_agents = []
        self.memories = []
        for params in agent_params:
            actor = StochasticActor(obs_dim=params.obs_dim, action_dim=params.action_dim, hidden_dim=params.hidden_dim)
            memory = Memory()
            ppo_agent = PpoAgent(
                actor=actor,
                critic=self.centralized_critic,
                memory=memory,
                learning_rate=params.learning_rate,
                gamma=params.gamma,
                epsilon=params.epsilon,
                K_epochs=params.K_epochs
            )
            self.ppo_agents.append(ppo_agent)