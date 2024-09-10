from pettingzoo.mpe import simple_spread_v3

from src.agents.q.vdn_agent import VdnAgent
from src.tests.common.common import LoopParams
from src.tests.common.simple_spread import SimpleSpread, SimpleSpreadParams


def main(simple_spread_params: SimpleSpreadParams, loop_params: LoopParams):
    simple_spread = SimpleSpread(params=simple_spread_params)
    simple_spread.reset()

    obs_dim, action_dim, n_agents = \
        simple_spread.obs_size, simple_spread.action_size, simple_spread.n_agents

    agent = VdnAgent(
        n_agents=n_agents,
        state_dim=obs_dim,
        hidden_dim=128,
        hidden_output_dim=32,
        n_actions=action_dim,
        learning_rate=0.000001,
        epsilon=0.1,
        gamma=0.99,
        buffer_capacity=10000,
        batch_size=20,
    )

    for episode in range(loop_params.max_episodes):
        simple_spread.reset()
        total_reward = 0
        step = 0

        while step < loop_params.max_timesteps:
            rewards, dones = agent.step(env=simple_spread.get_env(), step=step)

            loss = agent.update()

            total_reward += sum(rewards)
            step += 1

            if loss is not None:
                print(f"Episode {episode + 1}, Step {step + 1}, Total Reward: {total_reward}, Loss: {loss:.4f}")
            else:
                print(f"Episode {episode + 1}, Step {step + 1}, Total Reward: {total_reward}")

            if all(dones) and len(dones) != 0:
                break


        print(f"Episode {episode + 1}, Total Reward: {total_reward:.2f}")

if __name__ == "__main__":
    simple_spread_params = SimpleSpreadParams(n=3, local_ratio=0.5, max_cycles=25)
    loop_params = LoopParams(max_episodes=100, max_timesteps=1000, update_timestep=100)
    main(simple_spread_params=simple_spread_params, loop_params=loop_params)
