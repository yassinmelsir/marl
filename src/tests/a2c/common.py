from dataclasses import dataclass
from typing import Union, Type

from src.agents.a2c.ia2c_agent import Ia2cAgent
from src.agents.a2c.maa2c_agent import Maa2cAgent


@dataclass
class A2cParams:
    lr: float
    gamma: float
    eps_clip: float
    K_epochs: int
    hidden_dim: int
    entropy_coefficient: float
    agent: Type[Union[Ia2cAgent, Maa2cAgent]]