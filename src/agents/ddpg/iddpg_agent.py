from typing import Union

import torch
from src.agents.ddpg.ddpg_agent import DdpgAgent
from src.common.replay_buffer import ReplayBuffer
from src.networks.gumbel_actor import GumbelActor
from src.networks.value_critic import ValueCritic


class IddpgAgent:
    def __init__(
            self, n_agents: int, obs_dim: int, action_dim: int, hidden_dim: int, lr: float, gamma: float,
            eps_clip: float, K_epochs: int, buffer_size: int, batch_size: int, noise_scale: Union[float, None], temperature: float
    ):
        self.agents = []
        for _ in range(n_agents):
            actor = GumbelActor(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim, temperature=temperature)
            critic = ValueCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)
            target_actor = GumbelActor(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim,
                                       temperature=temperature)
            target_critic = ValueCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)
            replay_buffer = ReplayBuffer(batch_size=batch_size, buffer_size=buffer_size)
            ddpg_agent = DdpgAgent(
                actor=actor,
                critic=critic,
                target_actor=target_actor,
                target_critic=target_critic,
                replay_buffer=replay_buffer,
                lr=lr,
                gamma=gamma,
                eps_clip=eps_clip,
                K_epochs=K_epochs,
                noise_scale=noise_scale
            )
            self.agents.append(ddpg_agent)
            self.replay_buffer = ReplayBuffer(batch_size=batch_size, buffer_size=buffer_size)

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

    def step(self, env):
        states = []
        next_states = []
        rewards = []
        dones = []
        actions = []
        action_probs = []

        for idx, agent_id in enumerate(env.agents):
            observation, reward, termination, truncation, _ = env.last()
            obs_tensor = torch.FloatTensor(observation)

            if termination or truncation:
                return rewards, [True]
            else:
                action, action_probs_tensor = self.agents[idx].select_action(observation=obs_tensor)

            env.step(action)
            next_observation = env.observe(agent_id)

            next_obs_tensor = torch.FloatTensor(next_observation)
            action_tensor = torch.IntTensor([action])
            done_tensor = torch.BoolTensor([termination or truncation])
            reward_tensor = torch.FloatTensor([reward])

            states.append(obs_tensor)
            next_states.append(next_obs_tensor)
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

            self.agents[idx].replay_buffer.add(experience)

        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.stack(actions)
        action_probs = torch.stack(action_probs)
        rewards = torch.tensor(rewards)
        dones = torch.tensor(dones)

        experience = (
            states,
            next_states,
            actions,
            action_probs,
            rewards,
            dones
        )

        self.replay_buffer.add(experience)

        return rewards, dones

    def update(self):
        for idx, agent in enumerate(self.agents):
            agent.update()