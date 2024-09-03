from dataclasses import dataclass
from typing import Union, Type

from src.agents.ippo_agent import IppoAgent
from src.agents.mappo_agent import MappoAgent


@dataclass
class PpoParams:
    lr: float
    gamma: float
    eps_clip: float
    K_epochs: int
    hidden_dim: int
    agent: Type[Union[IppoAgent, MappoAgent]]