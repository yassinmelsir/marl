from src.agents.common import AgentParams
from src.agents.i_agent import IAgent
from src.agents.ppo.ppo_agent import PpoAgent
from src.common.memory import Memory
from src.networks.stochastic_actor import StochasticActor
from src.networks.state_critic import StateCritic


class IppoAgent(IAgent):
    def __init__(self, agent_params: list[AgentParams]):
        super().__init__(agent_params)
        self.agents = []
        for param in agent_params:
            actor = StochasticActor(obs_dim=param.obs_dim, action_dim=param.action_dim, hidden_dim=param.hidden_dim)
            critic = StateCritic(obs_dim=param.obs_dim, hidden_dim=param.hidden_dim)
            memory = Memory()
            agent = PpoAgent(
                actor=actor,
                critic=critic,
                memory=memory,
                learning_rate=param.learning_rate,
                gamma=param.gamma,
                epsilon=param.epsilon,
                K_epochs=param.K_epochs
            )
            self.agents.append(agent)


