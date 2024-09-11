from typing import Union

import torch
from src.agents.ddpg.ddpg_agent import DdpgAgent
from src.agents.ib_agent import IbAgent
from src.common.replay_buffer import ReplayBuffer
from src.networks.gumbel_actor import GumbelActor
from src.networks.value_critic import ValueCritic


class IddpgAgent(IbAgent):
    def __init__(
            self, buffer_size, batch_size, n_agents: int, obs_dim: int, action_dim: int, hidden_dim: int, lr: float, gamma: float,
            eps_clip: float, K_epochs: int, noise_scale: Union[float, None], temperature: float
    ):
        super().__init__(buffer_size, batch_size)
        self.agents = []
        for _ in range(n_agents):
            actor = GumbelActor(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim, temperature=temperature)
            critic = ValueCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)
            target_actor = GumbelActor(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim,
                                       temperature=temperature)
            target_critic = ValueCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)
            replay_buffer = ReplayBuffer(batch_size=batch_size, buffer_size=buffer_size)
            ddpg_agent = DdpgAgent(
                actor=actor,
                critic=critic,
                target_actor=target_actor,
                target_critic=target_critic,
                replay_buffer=replay_buffer,
                lr=lr,
                gamma=gamma,
                eps_clip=eps_clip,
                K_epochs=K_epochs,
                noise_scale=noise_scale
            )
            self.agents.append(ddpg_agent)
            self.replay_buffer = ReplayBuffer(batch_size=batch_size, buffer_size=buffer_size)