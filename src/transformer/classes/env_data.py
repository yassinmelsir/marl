from typing import Union

import numpy as np
import torch
from pettingzoo.mpe import simple_spread_v3

from marl.src.environment.traffic_environment import TrafficEnvironment


class EnvData:

    def __init__(self, data_filepath, num_agents, num_runs, steps_per_run, env, num_heads):
        self.data_filepath = data_filepath
        self.num_agents = num_agents
        self.num_runs = num_runs
        self.steps_per_run = steps_per_run
        self.num_heads = num_heads

        self.env = env
        self.raw_data = None
        self.clean_data = None
        self.traffic_env = isinstance(self.env, TrafficEnvironment)

    def gather_data(self):

        self.env.reset()

        data = []

        for run in range(self.num_runs):
            self.env.reset()
            run_data = []

            for step in range(self.steps_per_run):
                actions = {agent: self.env.action_space(agent).sample() for agent in self.env.agents}

                if self.traffic_env:
                    rewards, dones, observations, actions = list(self.env.step(actions))
                    run_data.append([observations, actions])

                else:
                    step_result = list(self.env.step(actions))

                    run_data.append(step_result)

                    dones = [value for _, value in step_result[3].items()]
                    rewards = [value for _, value in step_result[2].items()]

                print(f"Run: {run}. Step: {step}. Rewards: {sum(rewards)}. Dones: {all(dones)}")

                if all(dones):
                    break

            if len(run_data) > 0:
                data.append(run_data)

        self.raw_data = data


    def shape_data(self):
        clean_data = [[[] for _ in range(len(self.raw_data[-1]))] for _ in range(len(self.raw_data))]
        for i in range(len(self.raw_data)):
            for j in range(len(self.raw_data[i])):
                timestep_data = self.raw_data[i][j]
                if self.traffic_env:
                    obs = np.concatenate(timestep_data[0])
                    actions = np.concatenate(timestep_data[1])
                else:
                    obs = np.concatenate([v for k, v in timestep_data[0].items()])
                    actions = [v for k, v in timestep_data[1].items()]
                cat_obs_len = obs.shape[0] + actions.shape[0]
                padding = np.zeros(self.num_heads - cat_obs_len % self.num_heads)
                clean_data[i][j] = np.concatenate((obs, actions, padding))

        self.clean_data = np.array(clean_data)
        return self.clean_data

    def save_clean_data(self):
        np.save(self.data_filepath, self.clean_data)

