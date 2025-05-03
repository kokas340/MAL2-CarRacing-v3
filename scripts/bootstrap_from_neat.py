# scripts/bootstrap_from_neat.py
import pickle
import gymnasium as gym
import torch
import numpy as np
from agents.ppo_policy import CustomPolicy
from stable_baselines3 import PPO
from agents.neat_agent import NeatAgent

# Load NEAT genome
with open("best_genome.pkl", "rb") as f:
    genome = pickle.load(f)

import neat
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    "neat_config.txt"
)

env = gym.make("CarRacing-v3", render_mode="rgb_array")
agent = NeatAgent(genome, config)

# Create PPO model with same obs and action space
model = PPO(CustomPolicy, env, verbose=1)

# Bootstrap training data (supervised)
obs_list, action_list = [], []
obs, _ = env.reset()
for _ in range(5000):
    action = agent.act(obs)
    obs_list.append(obs.flatten())
    action_list.append(action)
    obs, _, done, truncated, _ = env.step(action)
    if done or truncated:
        obs, _ = env.reset()

# Convert to tensors
X = torch.tensor(np.array(obs_list), dtype=torch.float32)
Y = torch.tensor(np.array(action_list), dtype=torch.float32)

# Train policy network manually (supervised pretraining)
optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-4)
loss_fn = torch.nn.MSELoss()

print("📚 Pretraining PPO model on NEAT agent's behavior...")
for _ in range(20):
    pred = model.policy.actor(model.policy.extract_features(X))
    loss = loss_fn(pred, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Loss: {loss.item():.4f}")

model.save("ppo_bootstrapped_from_neat")
print("✅ Saved PPO model initialized from NEAT!")
