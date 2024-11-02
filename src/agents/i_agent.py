from typing import Union, Optional

import torch
from src.agents.common import AgentParams, CentralParams

class IAgent:
    def __init__(self, agent_params: list[AgentParams], central_params: Optional[CentralParams] = None):
        self.agent_params = agent_params
        self.agents = []

        self.replay_buffer = None if central_params is None else central_params.replay_buffer

    def update(self):
        for idx, agent in enumerate(self.agents):
            agent.update()
            if hasattr(agent, 'memory') and agent.memory is not None:
                agent.memory.clear_memory()

    def get_batch(self):
        batch = self.replay_buffer.sample()
        observations, next_observations, actions, action_probs, rewards, dones = zip(*batch)

        return (
            torch.stack(observations),
            torch.stack(next_observations),
            torch.stack(actions),
            torch.stack(action_probs),
            torch.stack(rewards),
            torch.stack(dones)
        )

    def get_memories(self):
        return [agent.memory for agent in self.agents]

    def save_agent_data(self, global_experience, agent_experience, agent):
        obs_tensor, next_observation, action, action_probs_tensor, reward, done = agent_experience
        observations, next_observations, actions, action_probs, rewards, dones = global_experience

        next_obs_tensor = torch.FloatTensor(next_observation)
        action_tensor = torch.IntTensor([action])
        done_tensor = torch.BoolTensor([done])
        reward_tensor = torch.FloatTensor([reward])

        observations.append(obs_tensor)
        next_observations.append(next_obs_tensor)
        actions.append(action_tensor)
        action_probs.append(action_probs_tensor)
        rewards.append(reward_tensor)
        dones.append(done_tensor)

        experience = (
            obs_tensor,
            next_obs_tensor,
            action_tensor,
            action_probs_tensor,
            reward_tensor,
            done_tensor
        )

        if hasattr(agent, 'memory') and agent.memory is not None:
            agent.memory.observations.append(obs_tensor)
            agent.memory.next_observations.append(next_obs_tensor)
            agent.memory.actions.append(action_tensor)
            agent.memory.action_probs.append(action_probs_tensor)
            agent.memory.rewards.append(reward_tensor)
            agent.memory.dones.append(done_tensor)
        elif hasattr(agent, 'replay_buffer') and agent.replay_buffer is not None:
            agent.replay_buffer.add(experience)
        else:
            raise "Error! No memory or replay buffer!"

    def save_global_data(self, experience):
        observations, next_observations, actions, action_probs, rewards, dones = experience

        observations = torch.stack(observations)
        next_observations = torch.stack(next_observations)
        actions = torch.stack(actions)
        action_probs = torch.stack(action_probs)
        rewards = torch.tensor(rewards)
        dones = torch.tensor(dones)

        experience = (
            observations,
            next_observations,
            actions,
            action_probs,
            rewards,
            dones
        )

        if self.replay_buffer is not None:
            self.replay_buffer.add(experience)

        return rewards, dones

