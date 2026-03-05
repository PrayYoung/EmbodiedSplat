import numpy as np

from embodied_splat_sim.envs.splat_nav_env import SplatNavEnv


def test_env_api_dummy():
    env = SplatNavEnv(
        config_path="configs/env.yaml",
        renderer_mode="dummy",
        max_steps=20,
        seed=0,
    )

    obs, info = env.reset(seed=0)
    assert isinstance(info, dict)
    assert "distance_to_goal" in info
    assert "collision" in info
    assert env.observation_space.contains(obs)

    for _ in range(20):
        action = env.action_space.sample()
        step = env.step(action)
        assert len(step) == 5
        obs, reward, terminated, truncated, info = step
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert "distance_to_goal" in info
        assert "collision" in info
        assert env.observation_space.contains(obs)

    # dtype/shape sanity
    assert obs.dtype == np.uint8
    assert obs.shape == env.observation_space.shape
