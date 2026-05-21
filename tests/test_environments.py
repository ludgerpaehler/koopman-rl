import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import koopmanrl.environments  # noqa: F401 — register custom environments

ENVS = ["LinearSystem-v0", "FluidFlow-v0", "Lorenz-v0", "DoubleWell-v0"]


@pytest.mark.parametrize("env_id", ENVS)
def test_env_gymnasium_api(env_id):
    env = gym.make(env_id)
    check_env(env.unwrapped, skip_render_check=True)
    env.close()


@pytest.mark.parametrize("env_id", ENVS)
def test_env_reset_step_cpu(env_id):
    env = gym.make(env_id)
    obs, info = env.reset(seed=0)
    assert isinstance(info, dict)
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float64

    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, (float, np.floating))
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert isinstance(info, dict)
        if terminated or truncated:
            obs, info = env.reset(seed=1)
            break

    env.close()
