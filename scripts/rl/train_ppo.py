from embodied_splat_sim.envs.splat_nav_env import SplatNavEnv

from stable_baselines3 import PPO

def main():
    env = SplatNavEnv(config_path="configs/env.yaml", renderer_mode="dummy")
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="runs/rl/tb")
    model.learn(total_timesteps=50_000)
    model.save("runs/rl/ppo_splatnav")

if __name__ == "__main__":
    main()
