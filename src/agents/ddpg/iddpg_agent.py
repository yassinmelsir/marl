from src.agents.ddpg.ddpg_agent import DdpgAgent
from src.agents.i_agent import IAgent
from src.common.replay_buffer import ReplayBuffer
from src.networks.gumbel_actor import GumbelActor
from src.networks.value_critic import ValueCritic


class IddpgAgent(IAgent):
    def __init__(self, agent_params):
        super().__init__(agent_params=agent_params)
        for param in agent_params:
            actor = GumbelActor(obs_dim=param.obs_dim, action_dim=param.action_dim, hidden_dim=param.hidden_dim, temperature=param.temperature)
            critic = ValueCritic(obs_dim=param.obs_dim, hidden_dim=param.hidden_dim)
            target_actor = GumbelActor(obs_dim=param.obs_dim, action_dim=param.action_dim, hidden_dim=param.hidden_dim,
                                       temperature=param.temperature)
            target_critic = ValueCritic(obs_dim=param.obs_dim, hidden_dim=param.hidden_dim)
            replay_buffer = ReplayBuffer(batch_size=param.batch_size, buffer_capacity=param.buffer_capacity)
            ddpg_agent = DdpgAgent(
                actor=actor,
                critic=critic,
                target_actor=target_actor,
                target_critic=target_critic,
                replay_buffer=replay_buffer,
                learning_rate=param.learning_rate,
                gamma=param.gamma,
                epsilon=param.epsilon,
                K_epochs=param.K_epochs,
                noise_scale=param.noise_scale
            )
            self.agents.append(ddpg_agent)
            self.replay_buffer = ReplayBuffer(batch_size=param.batch_size, buffer_capacity=param.buffer_capacity)