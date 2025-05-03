import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

# Create the environment with rendering and seeding
def make_env():
    def _init():
        env = gym.make("CarRacing-v3", render_mode="human")
        env.reset(seed=42)
        return env
    return _init

# Wrap the environment
env = DummyVecEnv([make_env()])
env = VecTransposeImage(env)

# Load pretrained model
model = PPO.load("ppo_bootstrapped_from_neat", env=env)

# Continue training
model.learn(total_timesteps=100_000)
model.save("ppo_finetuned_from_neat")
