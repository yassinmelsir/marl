from torch import optim
from src.agents.i_agent import IAgent
from src.agents.q.dqn_agent import DqnAgent
from src.common.replay_buffer import ReplayBuffer
from src.networks.deep_q_network import DeepQNetwork


class IdqnAgent(IAgent):
    def __init__(self, agent_params):
        super().__init__(agent_params=agent_params)
        for param in agent_params:
            q_network = DeepQNetwork(
                obs_dim=param.obs_dim,
                hidden_dim=param.hidden_dim,
                hidden_output_dim=param.hidden_output_dim,
                action_dim=param.action_dim
            )
            target_q_network = DeepQNetwork(
                obs_dim=param.obs_dim,
                hidden_dim=param.hidden_dim,
                hidden_output_dim=param.hidden_output_dim,
                action_dim=param.action_dim
            )

            target_q_network.load_state_dict(q_network.state_dict())
            optimizer = optim.Adam(q_network.parameters(), lr=param.learning_rate)
            replay_buffer = ReplayBuffer(
                batch_size=param.batch_size,
                buffer_capacity=param.buffer_capacity
            )
            agent = DqnAgent(
                q_network=q_network,
                target_q_network=target_q_network,
                optimizer=optimizer,
                replay_buffer=replay_buffer,
                epsilon=param.epsilon,
                gamma=param.gamma,
                action_dim=param.action_dim
            )
            self.agents.append(agent)