# train_ppo_from_neat.py
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn
import pickle
import cv2

# --- Load NEAT Agent ---
with open("best_genome.pkl", "rb") as f:
    best_genome = pickle.load(f)

import neat
config_path = "neat_config.txt"
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path
)

neat_net = neat.nn.FeedForwardNetwork.create(best_genome, config)

def preprocess(obs):
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (16, 12))
    normalized = resized.flatten() / 255.0
    extras = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return np.concatenate([normalized, extras])

def generate_dataset(env, neat_net, episodes=20):
    X, y = [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        while not done and not truncated:
            x = preprocess(obs)
            output = neat_net.activate(x)
            action = np.array([
                np.clip(output[0], -1.0, 1.0),
                np.clip(output[1],  0.0, 1.0),
                np.clip(output[2],  0.0, 1.0)
            ], dtype=np.float32)
            X.append(obs)
            y.append(action)
            obs, _, done, truncated, _ = env.step(action)
    return np.array(X), np.array(y)

# --- Custom feature extractor ---
class SimpleCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, obs):
        return self.linear(self.cnn(obs))

# --- Custom policy using our CNN ---
class CustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.actor = self.action_net  # expose actor explicitly
        self.critic = self.value_net  # (optional, if needed)


# --- Env & Model Setup ---
env = make_vec_env("CarRacing-v3", n_envs=1)
model = PPO(
    CustomPolicy,
    env,
    verbose=1,
    policy_kwargs=dict(
        features_extractor_class=SimpleCNN,
        features_extractor_kwargs=dict(features_dim=64)
    )
)



# --- Pretraining Dataset (images + NEAT actions) ---
print("🎯 Generating dataset from NEAT agent...")
demo_env = gym.make("CarRacing-v3")
X, y = generate_dataset(demo_env, neat_net, episodes=10)
demo_env.close()

# --- Imitation Learning (supervised pretraining) ---
print("🧠 Pretraining PPO with NEAT data...")
from torch.utils.data import TensorDataset, DataLoader
X_tensor = torch.tensor(X).permute(0, 3, 1, 2).float()
y_tensor = torch.tensor(y).float()
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-4)
model.policy.train()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.policy.to(device)

for epoch in range(3):
    losses = []
    for batch_obs, batch_act in loader:
        batch_obs = batch_obs.to(device)
        batch_act = batch_act.to(device)

        features = model.policy.extract_features(batch_obs)
        pred = model.policy.action_net(features)

        loss = ((pred - batch_act)**2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print(f"Epoch {epoch+1} Loss: {np.mean(losses):.4f}")

# --- Save & Train ---
model.save("ppo_bootstrapped_from_neat")
print("✅ PPO model saved. You can now run PPO fine-tuning on top.")
