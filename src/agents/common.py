from dataclasses import dataclass
from typing import Optional

@dataclass
class CentralParams:
    obs_dim: int
    hidden_dim: int
    learning_rate: int

@dataclass
class AgentParams:
    obs_dim: int
    action_dim: int
    hidden_dim: int
    learning_rate: float
    gamma: float
    epsilon: float
    K_epochs: int
    entropy_coefficient: Optional[float] = None