import torch

from src.agents.ppo.ppo_agent import PpoAgent
from src.common.memory import Memory
from src.networks.stochastic_actor import StochasticActor
from src.networks.state_critic import StateCritic


class IppoAgent:
    def __init__(self, n_agents, obs_dim, action_dim, hidden_dim, lr, gamma, eps_clip, K_epochs):
        self.ppo_agents = []
        for _ in range(n_agents):
            actor = StochasticActor(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)
            critic = StateCritic(obs_dim=obs_dim, hidden_dim=hidden_dim)
            memory = Memory()
            ppo_agent = PpoAgent(
                actor=actor,
                critic=critic,
                memory=memory,
                lr=lr,
                gamma=gamma,
                eps_clip=eps_clip,
                K_epochs=K_epochs
            )
            self.ppo_agents.append(ppo_agent)

    def step(self, env):
        dones = []
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

            self.ppo_agents[i].memory.log_probs.append(log_prob)
            self.ppo_agents[i].memory.actions.append(action)

            done = torch.BoolTensor([termination or truncation]).squeeze()
            reward = torch.FloatTensor([reward]).squeeze()
            obs_tensor = obs_tensor.squeeze()

            self.ppo_agents[i].memory.observations.append(obs_tensor)
            self.ppo_agents[i].memory.rewards.append(reward)
            self.ppo_agents[i].memory.dones.append(done)

            dones.append(termination or truncation)

        return dones

    def update(self):
        for idx, agent in enumerate(self.ppo_agents):
            agent.update()
            agent.memory.clear_memory()

    def get_memories(self):
        return [agent.memory for agent in self.ppo_agents]

