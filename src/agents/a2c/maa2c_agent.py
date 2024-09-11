import torch
from torch import optim
import torch.nn.functional as F

from src.agents.a2c.a2c_agent import A2cAgent
from src.agents.a2c.ia2c_agent import Ia2cAgent
from src.common.memory import Memory
from src.networks.stochastic_actor import StochasticActor
from src.networks.state_critic import StateCritic


class Maa2cAgent(Ia2cAgent):
    def __init__(self, n_agents, obs_dim, action_dim, hidden_dim, learning_rate, gamma, epsilon, K_epochs, entropy_coefficient):
        super().__init__(n_agents, obs_dim, action_dim, hidden_dim, learning_rate, gamma, epsilon, K_epochs, entropy_coefficient)
        self.a2c_agents = []
        self.memories = []
        self.centralized_critic = StateCritic(obs_dim=obs_dim, hidden_dim=hidden_dim)
        self.centralized_critic_optimizer = optim.Adam(self.centralized_critic.parameters(), lr=learning_rate)
        for _ in range(n_agents):
            actor = StochasticActor(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)
            memory = Memory()
            a2c_agent = A2cAgent(
                actor=actor,
                critic=self.centralized_critic,
                memory=memory,
                learning_rate=learning_rate,
                gamma=gamma,
                epsilon=epsilon,
                K_epochs=K_epochs,
                entropy_coefficient=entropy_coefficient
            )
            self.a2c_agents.append(a2c_agent)

    def update_centralized_critic(self, observations, rewards):

        q_values = self.centralized_critic(observations)

        critic_loss = 0.5 * F.mse_loss(q_values, rewards)
        self.centralized_critic_optimizer.zero_grad()
        critic_loss.backward()
        self.centralized_critic_optimizer.step()

        return q_values

    def update(self):
        for idx, agent in enumerate(self.a2c_agents):
            rewards, old_observations, old_actions, next_observations = agent.get_update_data()

            observation_values = self.update_centralized_critic(
                observations=old_observations,
                rewards=rewards
            )

            next_observation_values = self.centralized_critic(next_observations)

            agent.update_actor(
                old_observations=old_observations,
                old_actions=old_actions,
                rewards=rewards,
                observation_values=observation_values,
                next_observation_values=next_observation_values
            )

            agent.memory.clear_memory()


