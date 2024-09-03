from dataclasses import dataclass

import gymnasium as gym
@dataclass
class LoopParams:
    max_episodes: int
    max_timesteps: int
    update_timestep: int
