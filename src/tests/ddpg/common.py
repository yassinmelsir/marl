from dataclasses import dataclass
from typing import Union, Type

from src.agents.ddpg.iddpg_agent import IddpgAgent
from src.agents.ddpg.maddpg_agent import MaddpgAgent
from src.tests.common.common import LoopParams
from src.tests.common.simple_spread import SimpleSpreadParams


@dataclass
class DdpgParams:
    lr: float
    gamma: float
    eps_clip: float
    K_epochs: int
    hidden_dim: int
    batch_size: int
    buffer_size: int
    noise_scale: Union[float, None]
    temperature: float
    agent: Type[Union[IddpgAgent, MaddpgAgent]]


loop_params = LoopParams(max_episodes=100, max_timesteps=1000, update_timestep=100)
simple_spread_params = SimpleSpreadParams(n=3, local_ratio=0.5, max_cycles=25)
ddpg_params = DdpgParams(
    agent=IddpgAgent,
    hidden_dim=256, lr=3e-6,
    gamma=0.99, eps_clip=0.2,
    K_epochs=4,
    batch_size=5,
    buffer_size=10000,
    noise_scale=None,
    temperature=1.0
)
