from embodied_splat_sim.envs.splat_nav_env import SplatNavEnv
from stable_baselines3 import PPO


def main():
    # Keep resolution small at first to reduce render cost.
    env = SplatNavEnv(
        config_path="configs/env.yaml",
        renderer_mode="nerfstudio",
        max_steps=100,
    )

    # Use smaller rollout to reduce memory and speed up early debug.
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        n_steps=128,
        batch_size=64,
        tensorboard_log="runs/rl/tb",
    )

    model.learn(total_timesteps=2_000)
    model.save("runs/rl/ppo_nerfstudio_smoketest")


if __name__ == "__main__":
    main()
