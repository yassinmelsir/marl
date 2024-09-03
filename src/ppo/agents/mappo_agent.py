import torch

from src.ppo.agents.ippo_agent import IppoAgent
from src.ppo.agents.ppo_agent import PpoAgent
from src.ppo.common.memory import Memory
from src.ppo.networks.actor import Actor
from src.ppo.networks.critic import Critic


class MappoAgent(IppoAgent):
    def __init__(self, n_agents, obs_dim, action_dim, hidden_dim, lr, gamma, eps_clip, K_epochs):
        super().__init__(n_agents, obs_dim, action_dim, hidden_dim, lr, gamma, eps_clip, K_epochs)
        self.ppo_agents = []
        self.memories = []
        global_obs_dim = obs_dim * n_agents
        self.centralized_critic = Critic(obs_dim=global_obs_dim, hidden_dim=hidden_dim)
        for _ in range(n_agents):
            actor = Actor(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)
            ppo_agent = PpoAgent(
                actor=actor,
                critic=self.centralized_critic,
                lr=lr,
                gamma=gamma,
                eps_clip=eps_clip,
                K_epochs=K_epochs
            )
            self.ppo_agents.append(ppo_agent)
            self.memories.append(Memory())

