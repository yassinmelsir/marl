import torch
from torch import optim
import torch.nn.functional as F

from src.agents.ppo.ippo_agent import IppoAgent
from src.agents.ppo.ppo_agent import PpoAgent
from src.common.memory import Memory
from src.networks.actor import Actor
from src.networks.critic import Critic


class MappoAgent(IppoAgent):
    def __init__(self, n_agents, obs_dim, action_dim, hidden_dim, lr, gamma, eps_clip, K_epochs):
        super().__init__(n_agents, obs_dim, action_dim, hidden_dim, lr, gamma, eps_clip, K_epochs)
        self.ppo_agents = []
        self.memories = []
        global_obs_dim = obs_dim * n_agents
        self.centralized_critic = Critic(obs_dim=global_obs_dim, hidden_dim=hidden_dim)
        self.centralized_critic_optimizer = optim.Adam(self.centralized_critic.parameters(), lr=lr)
        for _ in range(n_agents):
            actor = Actor(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)
            memory = Memory()
            ppo_agent = PpoAgent(
                actor=actor,
                critic=self.centralized_critic,
                memory=memory,
                lr=lr,
                gamma=gamma,
                eps_clip=eps_clip,
                K_epochs=K_epochs
            )
            self.ppo_agents.append(ppo_agent)

    def update_centralized_critic(self, global_old_observations, global_rewards):

        global_observation_values = self.centralized_critic(global_old_observations)

        critic_loss = 0.5 * F.mse_loss(global_observation_values, global_rewards)
        self.centralized_critic_optimizer.zero_grad()
        critic_loss.backward()
        self.centralized_critic_optimizer.step()

        return global_observation_values

    def update(self):
        global_rewards, global_old_observations, global_old_actions, global_old_log_probs = [], [], [], []
        for idx, agent in enumerate(self.ppo_agents):
            rewards, old_observations, old_actions, old_log_probs = agent.get_update_data()
            global_rewards.append(rewards)
            global_old_observations.append(old_observations)
            global_old_actions.append(old_actions)
            global_old_log_probs.append(old_log_probs)

        global_old_observations = torch.stack(global_old_observations)

        num_agents, timesteps, global_obs_dim = global_old_observations.shape
        global_old_observations = global_old_observations.view(timesteps, num_agents * global_obs_dim)

        global_rewards = torch.stack(global_rewards).sum(dim=0)

        global_observation_values = self.update_centralized_critic(
            global_old_observations=global_old_observations,
            global_rewards=global_rewards
        )

        for idx, agent in enumerate(self.ppo_agents):
            rewards, old_observations, old_actions, old_log_probs = agent.get_update_data()

            agent.update_actor(
                old_observations=old_observations,
                old_actions=old_actions,
                old_log_probs=old_log_probs,
                rewards=rewards,
                observation_values=global_observation_values
            )

            agent.memory.clear_memory()


