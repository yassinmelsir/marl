from typing import Union

import numpy as np
import torch
from pettingzoo.mpe import simple_spread_v3

class GatherTransformerData:

    def __init__(self, file_name, num_agents, num_runs, steps_per_run, env):
        self.file_name = file_name
        self.num_agents = num_agents
        self.num_runs = num_runs
        self.steps_per_run = steps_per_run

        self.env = env
        self.data = None


    def gather_data(self):

        self.env.reset()

        data = []

        for run in range(self.num_runs):
            self.env.reset()
            run_data = []

            for step in range(self.steps_per_run):
                actions = {agent: self.env.action_space(agent).sample() for agent in self.env.agents}

                step_result = list(self.env.step(actions))
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

        self.data = data

    def shape_data(self):
        for i in range(len(self.data)):
            for timestep in self.data[i]:
                observations = np.array(np.array([v for k, v in timestep[0].items()]).reshape(-1))
                actions = np.array([v for k, v in timestep[1].items()])
                # rewards = np.array([v for k, v in timestep_data[2].items()])
                # dones = np.array([v for k, v in timestep_data[3].items()])
                timestep = np.concatenate([observations, actions, np.zeros(3)])

                self.data[i] = np.array(timestep)

    def print_data(self):
        for i in range(len(self.data)):
            print(self.data[i])

    def save_data(self):
        np.save(self.file_name, self.data)