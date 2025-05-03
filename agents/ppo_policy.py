# agents/ppo_policy.py
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor

class NEATFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        x = self.flatten(observations)
        return self.fc(x)
    
class CustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            features_extractor_class=NEATFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=64),
            **kwargs
        )