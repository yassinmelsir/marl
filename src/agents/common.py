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
    buffer_capacity: Optional[int] = None
    batch_size: Optional[int] = None
    temperature: Optional[float] = None
    entropy_coefficient: Optional[float] = None
    noise_scale: Optional[float] = None
