import torch
from torch import optim
import torch.nn.functional as F
from src.agents.i_agent import IAgent
from src.networks.state_critic import StateCritic


class Maacc(IAgent):
    def __init__(self, agent_params, central_params):
        super().__init__(agent_params, central_params)
        self.ppo_agents = []
        self.memories = []
        self.centralized_critic = StateCritic(obs_dim=central_params.obs_dim, hidden_dim=central_params.hidden_dim)
        self.centralized_critic_optimizer = optim.Adam(self.centralized_critic.parameters(),
                                                       lr=central_params.learning_rate)

    def update_centralized_critic(self, global_observations, global_rewards):

        global_observation_values = self.centralized_critic(global_observations)

        critic_loss = 0.5 * F.mse_loss(global_observation_values, global_rewards)
        self.centralized_critic_optimizer.zero_grad()
        critic_loss.backward()
        self.centralized_critic_optimizer.step()

        return global_observation_values

    def update(self):
        global_rewards, global_observations, global_actions, global_action_probs = [], [], [], []
        for idx, agent in enumerate(self.ppo_agents):
            rewards, observations, actions, action_probs = agent.get_update_data()
            global_rewards.append(rewards)
            global_observations.append(observations)
            global_actions.append(actions)
            global_action_probs.append(action_probs)

        global_observations = torch.stack(global_observations)

        num_agents, timesteps, obs_dim = global_observations.shape
        global_observations = global_observations.view(timesteps, num_agents * obs_dim)

        global_rewards = torch.stack(global_rewards).sum(dim=0)

        global_observation_values = self.update_centralized_critic(
            global_observations=global_observations,
            global_rewards=global_rewards
        )

        for idx, agent in enumerate(self.ppo_agents):
            rewards, observations, actions, action_probs = agent.get_update_data()

            agent.update_actor(
                observations=observations,
                actions=actions,
                action_probs=action_probs,
                rewards=rewards,
                observation_values=global_observation_values
            )

            agent.memory.clear_memory()
