import cv2
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

# --- Environment Setup (no wrappers, no reward shaping) ---
def make_env():
    def _init():
        return gym.make("CarRacing-v3", render_mode="rgb_array", domain_randomize=False)
    return _init

env = DummyVecEnv([make_env()])
env = VecTransposeImage(env)

# --- Load Trained Model ---
model = PPO.load("ppo_finetuned_offtrack_box", env=env)

# --- Run Episode ---
obs = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _ = env.step(action)

    # Get rendered frame
    frame = env.envs[0].render()
    frame = cv2.resize(frame, (frame.shape[1] * 2, frame.shape[0] * 2))  # Optional zoom
    cv2.imshow("CarRacing PPO Agent", frame)

    # Quit with Q or end of episode
    if cv2.waitKey(1) & 0xFF == ord('q') or done[0]:
        break

cv2.destroyAllWindows()
