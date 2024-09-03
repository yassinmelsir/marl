import gymnasium as gym


def get_obs_action_size(env):
    first_agent = env.agents[0]
    observation_space = env.observation_space(first_agent)
    action_space = env.action_space(first_agent)

    if isinstance(observation_space, gym.spaces.Box):
        obs_size = observation_space.shape[0]
    else:
        raise ValueError(f"Unexpected observation space type: {type(observation_space)}")

    if isinstance(action_space, gym.spaces.Discrete):
        print(f"action_space: {action_space}")
        action_size = action_space.n
    else:
        raise ValueError(f"Unexpected action space type: {type(action_space)}")

    return obs_size, action_size