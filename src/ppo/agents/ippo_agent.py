import torch

from src.ppo.agents.ppo_agent import PpoAgent
from src.ppo.common.memory import Memory
from src.ppo.networks.actor import Actor
from src.ppo.networks.critic import Critic


class IppoAgent:
    def __init__(self, n_agents, obs_dim, action_dim, hidden_dim, lr, gamma, eps_clip, K_epochs):
        self.ppo_agents = []
        self.memories = []
        for _ in range(n_agents):
            actor = Actor(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)
            critic = Critic(obs_dim=obs_dim, hidden_dim=hidden_dim)
            ppo_agent = PpoAgent(
                actor=actor,
                critic=critic,
                lr=lr,
                gamma=gamma,
                eps_clip=eps_clip,
                K_epochs=K_epochs
            )
            self.ppo_agents.append(ppo_agent)
            self.memories.append(Memory())

    def step(self, env):
        for i, agent_id in enumerate(env.agents):
            observation, reward, termination, truncation, _ = env.last()
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)

            if termination or truncation:
                return [True]
            else:
                action, log_prob = self.ppo_agents[i].select_action(obs_tensor)
                log_prob = log_prob.squeeze()

            env.step(action)

            if action is not None:
                action = torch.IntTensor([action]).squeeze()

            self.memories[i].log_probs.append(log_prob)
            self.memories[i].actions.append(action)

            self.memories[i].observations.append(obs_tensor.squeeze())
            self.memories[i].rewards.append(torch.FloatTensor([reward]).squeeze())
            self.memories[i].dones.append(torch.BoolTensor([termination or truncation]).squeeze())

        return [row.dones[-1] for row in self.memories]

    def update(self):
        for idx, agent in enumerate(self.ppo_agents):
            agent.update(self.memories[idx])
            self.memories[idx].clear_memory()

    def get_memories(self):
        return self.memories

