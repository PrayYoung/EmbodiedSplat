from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from embodied_splat_sim.envs.splat_nav_env import SplatNavEnv


def make_env(config_path: str, renderer_mode: str, max_steps: int, seed: int):
    def _init():
        return SplatNavEnv(
            config_path=config_path,
            renderer_mode=renderer_mode,
            max_steps=max_steps,
            seed=seed,
        )

    return _init


def main():
    config_path = "configs/env.yaml"
    max_steps = 100
    n_envs = 4

    # Nerfstudio renderer is not fork-safe; use DummyVecEnv
    nerf_env = DummyVecEnv([make_env(config_path, "nerfstudio", max_steps, 0)])

    # Dummy renderer can scale with subprocess workers
    dummy_env = SubprocVecEnv([
        make_env(config_path, "dummy", max_steps, i) for i in range(n_envs)
    ])

    nerf_model = PPO(
        "CnnPolicy",
        nerf_env,
        verbose=1,
        n_steps=128,
        batch_size=64,
        tensorboard_log="runs/rl/tb",
    )
    nerf_model.learn(total_timesteps=2_000)
    nerf_model.save("runs/rl/ppo_nerfstudio_vec_smoketest")

    dummy_model = PPO(
        "CnnPolicy",
        dummy_env,
        verbose=1,
        n_steps=256,
        batch_size=128,
        tensorboard_log="runs/rl/tb",
    )
    dummy_model.learn(total_timesteps=5_000)
    dummy_model.save("runs/rl/ppo_dummy_vec_smoketest")


if __name__ == "__main__":
    main()
