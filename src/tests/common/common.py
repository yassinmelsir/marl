from dataclasses import dataclass

@dataclass
class LoopParams:
    max_episodes: int
    max_timesteps: int
    update_timestep: int
