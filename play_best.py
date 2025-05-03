import gymnasium as gym
import pickle
import neat
from agents.neat_agent import NeatAgent
from run_episode import run_agent

# Load the saved genome
with open("best_genome.pkl", "rb") as f:
    genome = pickle.load(f)

# Load NEAT config (same as used in evolve)
config_path = "neat_config.txt"
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path
)

# Wrap genome in NeatAgent
agent = NeatAgent(genome, config)

# Run with rendering ON
env = gym.make("CarRacing-v3", render_mode="human")
reward = run_agent(env, agent, render=True)
env.close()

print(f"🎮 Best agent scored: {reward:.2f}")
