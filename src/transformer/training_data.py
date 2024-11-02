import numpy as np
from pettingzoo.mpe import simple_spread_v3

file_name = "transfomer_training_data.npy"

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

        step_result = env.step(actions)
        run_data.append(step_result)

        dones = [value for _, value in step_result[3].items()]
        rewards = [value for _, value in step_result[2].items()]

        print(f"Run: {run}. Step: {step}. Rewards: {sum(rewards)}. Dones: {dones}")

        if all(dones):
            break

    data.append(run_data)

    print()

np.save(file_name, data)
