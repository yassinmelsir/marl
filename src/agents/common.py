from dataclasses import dataclass
from typing import Optional
from src.common.replay_buffer import ReplayBuffer


@dataclass
class CentralParams:
    obs_dim: int
    hidden_dim: int
    learning_rate: float
    replay_buffer: Optional[ReplayBuffer] = None
    gamma: Optional[float] = None
    action_dim: Optional[float] = None

@dataclass
class AgentParams:
    obs_dim: int
    action_dim: int
    hidden_dim: int
    learning_rate: float
    gamma: float
    epsilon: float
    K_epochs: int
    hidden_output_dim: Optional[int] = None
    buffer_capacity: Optional[int] = None
    batch_size: Optional[int] = None
    temperature: Optional[float] = None
    entropy_coefficient: Optional[float] = None
    noise_scale: Optional[float] = None