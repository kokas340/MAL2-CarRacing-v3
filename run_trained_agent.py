import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

# Load environment with rendering
def make_env():
    return gym.make("CarRacing-v3", render_mode="human")

env = DummyVecEnv([make_env])
env = VecTransposeImage(env)

# Load your trained model
model = PPO.load("ppo_finetuned_from_neat", env=env)

obs = env.reset()
done = False

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
