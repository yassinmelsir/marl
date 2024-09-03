import torch
from pettingzoo.mpe import simple_spread_v3
import numpy as np

from src.indep.ppo.i_agent import IPpoAgent
from src.tests.common import get_obs_action_size

env = simple_spread_v3.env(N=3, local_ratio=0.5, max_cycles=25)
env.reset()

obs_dim, action_dim = get_obs_action_size(env=env)

ippo_agent = IPpoAgent(
    n_agents=len(env.agents),
    obs_dim=obs_dim,
    action_dim=action_dim,
    lr=3e-4,
    gamma=0.99,
    eps_clip=0.2,
    K_epochs=4
)

max_episodes = 1000
max_timesteps = 100
update_timestep = 2000

timestep = 0

for episode in range(max_episodes):
    env.reset()
    for t in range(max_timesteps):

        dones = ippo_agent.step(env=env)

        if timestep % update_timestep == 0:
            ippo_agent.update()
            timestep = 0

        timestep += 1

        if all(dones):
            break

        if (timestep + 1) % 100 == 0:
            print(
                f"timestep {timestep + 1} - average reward: {np.mean([np.sum(m.rewards) for m in ippo_agent.get_memories()])}")

    print(f"Episode {episode + 1} finished")

    if (episode + 1) % 100 == 0:
        print(
            f"Episode {episode + 1} - average reward: {np.mean([np.sum(m.rewards) for m in ippo_agent.get_memories()])}")
