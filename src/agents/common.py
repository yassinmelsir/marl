from dataclasses import dataclass
from typing import Optional
from src.common.replay_buffer import ReplayBuffer


@dataclass
class CentralParams:
    obs_dim: int
    hidden_dim: int
    learning_rate: float
    replay_buffer: Optional[ReplayBuffer] = None

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
    replay_buffer: Optional[ReplayBuffer] = None
