import torch
from torch import optim
import torch.nn.functional as F

from src.agents.a2c.a2c_agent import A2cAgent
from src.agents.i_agent import IAgent
from src.agents.ppo.ippo_agent import IppoAgent
from src.common.memory import Memory
from src.networks.state_critic import StateCritic
from src.networks.stochastic_actor import StochasticActor


class Maacc(IAgent):
    def __init__(self, agent_params, central_params):
        super().__init__(agent_params, central_params)
        self.agents = []
        self.memories = []
        self.centralized_critic = StateCritic(obs_dim=central_params.obs_dim, hidden_dim=central_params.hidden_dim)
        self.centralized_critic_optimizer = optim.Adam(self.centralized_critic.parameters(),
                                                       lr=central_params.learning_rate)
        for param in agent_params:
            actor = StochasticActor(obs_dim=param.obs_dim, action_dim=param.action_dim, hidden_dim=param.hidden_dim)
            memory = Memory()
            agent = A2cAgent(
                actor=actor,
                critic=self.centralized_critic,
                memory=memory,
                learning_rate=param.learning_rate,
                gamma=param.gamma,
                epsilon=param.epsilon,
                K_epochs=param.K_epochs,
                entropy_coefficient=param.entropy_coefficient
            )
            self.agents.append(agent)


    def update_centralized_critic(self, global_observations, global_rewards):

        global_observation_values = self.centralized_critic(global_observations)

        critic_loss = 0.5 * F.mse_loss(global_observation_values, global_rewards)
        self.centralized_critic_optimizer.zero_grad()
        critic_loss.backward()
        self.centralized_critic_optimizer.step()

        return global_observation_values

    def update(self):
        global_rewards, global_observations, global_actions, global_action_probs = [], [], [], []
        for idx, agent in enumerate(self.agents):
            if len(agent.memory.observations) == 0:
                continue
            rewards, observations, actions, action_probs = agent.get_update_data()

            global_rewards.append(rewards)
            global_observations.append(observations)
            global_actions.append(actions)
            global_action_probs.append(action_probs)

        if len(global_observations) == 0: return

        global_observations = torch.stack(global_observations)

        num_agents, timesteps, obs_dim = global_observations.shape
        global_observations = global_observations.view(timesteps, num_agents * obs_dim)

        global_rewards = torch.stack(global_rewards).sum(dim=0)

        global_obs_values = self.update_centralized_critic(
            global_observations=global_observations,
            global_rewards=global_rewards
        )

        global_next_obs_values = self.centralized_critic(global_observations)

        for idx, agent in enumerate(self.agents):
            rewards, observations, actions, action_probs = agent.get_update_data()

            agent.update_actor(
                observations=observations,
                actions=actions,
                action_probs=action_probs,
                rewards=rewards,
                obs_values=global_obs_values,
                next_obs_values=global_next_obs_values,
            )

            agent.memory.clear_memory()
