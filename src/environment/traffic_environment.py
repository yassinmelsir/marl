import os
import random
import numpy as np
import torch
import traci
import subprocess
from dataclasses import dataclass


class ActionSpace:
    def __init__(self, actions):
        self.actions = actions

    def sample(self):
        action_space = self.actions
        return random.sample(action_space, k=1)

@dataclass
class TrafficEnvironmentParams:
    sumo_cmd: str
    gui: bool


class TrafficEnvironment:
    def __init__(self, gui, sumo_cmd, ai_control=False):
        self.gui = gui
        self.sumo_cmd = sumo_cmd.split()
        self.simulation_running = False
        self.traci = traci
        self.ai_control = ai_control

        self.start_simulation()
        self.agents = self.traci.trafficlight.getIDList()

    def _start_xquartz(self):
        result = subprocess.run(["open", "-a", "XQuartz"], capture_output=True, text=True)
        print("stdout:", result.stdout)
        print("stderr:", result.stderr)

    def start_simulation(self):
        if self.gui:
            self._start_xquartz()

        print("SUMO Command:", self.sumo_cmd)
        self.traci.start(self.sumo_cmd)
        self.simulation_running = True
        print(f"Simulation Running: {self.simulation_running}")

    def close_simulation(self):
        if self.simulation_running:
            self.traci.close()
            self.simulation_running = False
            print("Simulation completed and data logged.")
        else:
            print("Simulation was not running.")

    def get_agent_params(self):
        params = []
        for agent_id in self.agents:
            obs_dim = self.get_traci_traffic_light_observation(agent_id)
            action_dim = self.get_possible_actions(agent_id)
            params.append((obs_dim, action_dim))
        return params


    def get_env(self):
        return self.traci

    def reset(self):
        return

    def step(self, agent):
        observations = []
        next_observations = []
        rewards = []
        dones = []
        actions = []
        action_probs = []

        global_experience = (
            observations,
            next_observations,
            actions,
            action_probs,
            rewards,
            dones
        )

        for tl_id in self.agents:
            observation = self.get_traci_traffic_light_observation(tl_id)
            obs_tensor = torch.FloatTensor(observation)
            observations.append(obs_tensor)

            if not self.ai_control:
                action, action_probs_tensor = agent.agents[tl_id].select_action(observation=observations[tl_id])

                self.apply_traci_traffic_light_action(tl_id, action)
            else:
                light_program = self.traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)
                phase = self.traci.trafficlight.getPhase(tl_id)

                action = torch.tensor(np.array([phase]))
                action_probs_tensor = torch.tensor(np.zeros(len(light_program[0].phases)))

            actions.append(action)
            action_probs.append(action_probs_tensor)

        self.traci.simulationStep()

        for idx, tl_id in enumerate(self.agents):
            next_observation = self.get_traci_traffic_light_observation(tl_id)
            reward = self.get_traci_traffic_light_reward(tl_id)
            done = self.check_traci_traffic_light_done(tl_id)

            next_observations.append(torch.FloatTensor(next_observation))
            rewards.append(reward)
            dones.append(done)

            agent_experience = (
                observations[idx],
                next_observations[idx],
                actions[idx],
                action_probs[idx],
                rewards[idx],
                dones[idx]
            )

            if not self.ai_control:
                agent.save_agent_data(global_experience, agent_experience, agent=agent.agents[idx])


        if not self.ai_control:
            rewards, dones = agent.save_global_data(global_experience)

        return rewards, dones, observations, actions

    def get_traci_traffic_light_observation(self, tl_id):
        phase = self.traci.trafficlight.getPhase(tl_id)
        queue_lengths = [self.traci.lane.getLastStepVehicleNumber(lane_id) for lane_id in
                         self.traci.trafficlight.getControlledLanes(tl_id)]
        return [phase] + queue_lengths

    def apply_traci_traffic_light_action(self, tl_id, action):
        self.traci.trafficlight.setPhase(tl_id, action)

    def get_traci_traffic_light_reward(self, tl_id):
        queue_lengths = [self.traci.lane.getLastStepVehicleNumber(lane_id) for lane_id in
                         self.traci.trafficlight.getControlledLanes(tl_id)]
        queue_length_penalty = -sum(queue_lengths)

        emissions = [self.traci.vehicle.getCO2Emission(veh_id) for veh_id in self.traci.vehicle.getIDList()]
        emissions_penalty = -sum(emissions)

        delays = [self.traci.vehicle.getWaitingTime(veh_id) for veh_id in self.traci.vehicle.getIDList()]
        delay_penalty = -sum(delays)

        reward = queue_length_penalty + emissions_penalty + delay_penalty
        return reward

    def check_traci_traffic_light_done(self, tl_id):
        return False

    def get_possible_actions(self, tl_id):
        light_program = self.traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)

        num_phases = len(light_program[0].phases) if light_program else 0
        possible_actions = list(range(num_phases))

        return possible_actions

    def action_space(self, tl_id):
        return ActionSpace(actions=self.get_possible_actions(tl_id))



