import cv2
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from gymnasium import Wrapper
import imageio

# --- Custom Reward Wrapper ---
class CustomRewardWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        if action[1] > 0.5:
            reward += 2
        if action[2] > 0.3:
            reward -= 1.5
        return obs, reward, done, truncated, info

# --- Env setup ---
def make_env():
    def _init():
        env = gym.make("CarRacing-v3", render_mode="rgb_array")
        env = CustomRewardWrapper(env)
        return env
    return _init

# --- Prepare environment and model ---
frames = []
env = DummyVecEnv([make_env()])
env = VecTransposeImage(env)
model = PPO.load("ppo_finetuned_from_neat", env=env)

obs = env.reset()
total_reward = 0

# --- Unwrap env for direct rendering ---
base_env = env.envs[0]
render_env = base_env.unwrapped

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    total_reward += reward[0]

    frame = render_env.render()
    frame = cv2.resize(frame, (frame.shape[1] * 2, frame.shape[0] * 2))

    # Overlay reward
    cv2.putText(
        frame,
        f"Reward: {total_reward:.2f}",
        (50, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (139, 0, 0),  # Dark blue
        3,
        cv2.LINE_AA
    )

    # Convert to RGB for imageio and save for GIF
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(rgb_frame)

    # Display
    cv2.imshow("CarRacing - Reward Overlay", frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or done[0]:
        break

# Cleanup
cv2.destroyAllWindows()
print(f"✅ Episode finished with reward: {total_reward:.2f}")

# Save the GIF
gif_path = "ppo_run.gif"
imageio.mimsave(gif_path, frames, fps=30)
print(f"📽️  GIF saved to: {gif_path}")
