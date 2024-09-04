import torch
from torch import optim
import torch.nn.functional as F

from src.agents.a2c.a2c_agent import A2cAgent
from src.agents.a2c.ia2c_agent import Ia2cAgent
from src.common.memory import Memory
from src.networks.stochastic_actor import Actor
from src.networks.state_critic import Critic


class Maa2cAgent(Ia2cAgent):
    def __init__(self, n_agents, obs_dim, action_dim, hidden_dim, lr, gamma, eps_clip, K_epochs, entropy_coefficient):
        super().__init__(n_agents, obs_dim, action_dim, hidden_dim, lr, gamma, eps_clip, K_epochs, entropy_coefficient)
        self.a2c_agents = []
        self.memories = []
        self.centralized_critic = Critic(obs_dim=obs_dim, hidden_dim=hidden_dim)
        self.centralized_critic_optimizer = optim.Adam(self.centralized_critic.parameters(), lr=lr)
        for _ in range(n_agents):
            actor = Actor(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)
            memory = Memory()
            a2c_agent = A2cAgent(
                actor=actor,
                critic=self.centralized_critic,
                memory=memory,
                lr=lr,
                gamma=gamma,
                eps_clip=eps_clip,
                K_epochs=K_epochs,
                entropy_coefficient=entropy_coefficient
            )
            self.a2c_agents.append(a2c_agent)

    def update_centralized_critic(self, agent_observations, agent_rewards):

        observation_values = self.centralized_critic(agent_observations)

        critic_loss = 0.5 * F.mse_loss(observation_values, agent_rewards)
        self.centralized_critic_optimizer.zero_grad()
        critic_loss.backward()
        self.centralized_critic_optimizer.step()

        return observation_values

    def update(self):
        for idx, agent in enumerate(self.a2c_agents):
            rewards, old_observations, old_actions, next_observations = agent.get_update_data()

            observation_values = self.update_centralized_critic(
                agent_observations=old_observations,
                agent_rewards=rewards
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


