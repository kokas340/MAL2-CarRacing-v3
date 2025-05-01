import gymnasium as gym

# Create the CarRacing environment
env = gym.make("CarRacing-v3", render_mode="human")

obs, info = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # Random action
    obs, reward, done, truncated, info = env.step(action)

env.close()
