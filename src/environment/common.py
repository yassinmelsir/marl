from dataclasses import dataclass


@dataclass
class LoopParams:
    max_timesteps: int
    max_episodes: int
    update_timestep: int