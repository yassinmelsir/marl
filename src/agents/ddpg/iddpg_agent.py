import torch
from src.agents.ddpg.ddpg_agent import DdpgAgent
from src.common.replay_buffer import ReplayBuffer
from src.networks.gumbel_actor import GumbelActor
from src.networks.value_critic import ValueCritic


class IddpgAgent:
    def __init__(
            self, n_agents: int, obs_dim: int, action_dim: int, hidden_dim: int, lr: float, gamma: float,
            eps_clip: float, K_epochs: int, buffer_size: int, batch_size: int, noise_scale: float, temperature: float
    ):
        self.ddpg_agents = []
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
            self.ddpg_agents.append(ddpg_agent)

    def step(self, env) -> tuple[list, list]:
        dones = []
        rewards = []
        for i, agent_id in enumerate(env.agents):
            observation, reward, termination, truncation, _ = env.last()
            obs_tensor = torch.FloatTensor(observation)

            if termination or truncation:
                return rewards, [True]
            else:
                action, action_probs_tensor = self.ddpg_agents[i].select_action(obs_tensor)

            env.step(action)
            next_observation = env.observe(agent_id)

            next_obs_tensor = torch.FloatTensor(next_observation)
            done_tensor = torch.BoolTensor([termination or truncation])
            action_tensor = torch.IntTensor([action])
            reward_tensor = torch.FloatTensor([reward])

            experience = (
                obs_tensor,
                next_obs_tensor,
                action_tensor,
                action_probs_tensor,
                reward_tensor,
                done_tensor,
            )

            assert len(experience) == 6

            self.ddpg_agents[i].replay_buffer.add(experience)

            dones.append(termination or truncation)
            rewards.append(reward)

        return rewards, dones

    def update(self):
        for idx, agent in enumerate(self.ddpg_agents):
            agent.update()