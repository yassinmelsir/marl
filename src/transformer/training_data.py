import numpy as np
import torch
from pettingzoo.mpe import simple_spread_v3

file_name = "data/transfomer_training_data.npy"

num_agents = 3
num_runs = 100
steps_per_run = 50

env = simple_spread_v3.parallel_env(N=num_agents)
env.reset()

data = []

for run in range(num_runs):
    env.reset()
    run_data = []
    init_env_agents = env.agents
    for step in range(steps_per_run):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}

        step_result = list(env.step(actions))
        step_result[-1] = actions

        run_data.append(step_result)

        dones = [value for _, value in step_result[3].items()]
        rewards = [value for _, value in step_result[2].items()]

        print(f"Run: {run}. Step: {step}. Rewards: {sum(rewards)}. Dones: {dones}")

        if all(dones):
            break

    data.append(run_data)

    print(run_data[-2])
    print(run_data[-1])


def process_timestep(timestep_data):
    observations = np.array(np.array([v for k, v in timestep_data[0].items()]).reshape(-1))
    actions = np.array([v for k, v in timestep_data[1].items()])
    rewards = np.array([v for k, v in timestep_data[2].items()])
    dones = np.array([v for k, v in timestep_data[3].items()])
    timestep = np.concatenate([observations, actions, rewards, dones, [0]])
    return timestep

for i in range(len(data)):
    data[i] = np.array([process_timestep(timestep) for timestep in data[i]])

for i in range(len(data)):
    print(data[-1])

np.save(file_name, data)
