import sys
import os
import pickle
import neat
import gymnasium as gym
import matplotlib.pyplot as plt
from agents.neat_agent import NeatAgent
from run_episode import run_agent
import numpy as np  # 🔹 Needed for standard deviation

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'neat_config.txt')
generation_counter = 0 

fitness_history = []
diversity_history = []   # 🔹 New: for fitness std dev
species_count_history = []  # 🔹 New: for species tracking

# --- Evaluate a single genome ---
def eval_genome(genome, config):
    env = gym.make("CarRacing-v3", render_mode="none")
    obs, _ = env.reset(seed=42)
    agent = NeatAgent(genome, config)
    score, _, _, _ = run_agent(env, agent, render=False) 
    env.close()
    return score

# --- Evaluate a whole generation ---
def eval_genomes(genomes, config):
    global generation_counter
    print(f"\n🚀 Evaluating Generation {generation_counter} with {len(genomes)} genomes")

    fitnesses = []
    best_fitness = float("-inf")
    total_fitness = 0

    for genome_id, genome in genomes:
        fitness = eval_genome(genome, config)
        genome.fitness = fitness
        fitnesses.append(fitness)
        total_fitness += fitness
        if fitness > best_fitness:
            best_fitness = fitness
        print(f"  🧬 Genome {genome_id:>4} | Fitness: {fitness:7.2f}")

    avg_fitness = total_fitness / len(genomes)
    std_fitness = np.std(fitnesses)  # 🔹 Track diversity
    fitness_history.append((generation_counter, best_fitness, avg_fitness))
    diversity_history.append(std_fitness)

    # 🔹 Track species count using config's species_set
    # 🔹 Safe species count tracking
    if generation_counter > 0:
        try:
            num_species = len(stats.get_species_sizes()[-1])
        except (IndexError, ValueError):
            num_species = 0
    else:
        num_species = 0

    species_count_history.append(num_species)

    print(f"📈 Gen {generation_counter} Summary | Best: {best_fitness:.2f} | Avg: {avg_fitness:.2f} | Std: {std_fitness:.2f} | Species: {num_species}")

    generation_counter += 1

if __name__ == "__main__":
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH
    )

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    winner = population.run(eval_genomes, n=100)

    print("\n🎉 Evolution complete!")
    print("Best genome:")
    print(winner)

    best_genome_path = f"best_genome_gen{generation_counter - 1}.pkl"
    with open(best_genome_path, "wb") as f:
        pickle.dump(winner, f)
    print(f"✅ Best genome saved to {best_genome_path}")

    # --- Plot fitness progress ---
    gens, bests, avgs = zip(*fitness_history)
    plt.figure(figsize=(10, 5))
    plt.plot(gens, bests, label="Best Fitness")
    plt.plot(gens, avgs, label="Average Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("NEAT Evolution Fitness Progress")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fitness_progress.png")
    plt.show()
    print("📊 Fitness progress saved")

    # --- 🔹 Plot fitness standard deviation (diversity) ---
    plt.figure(figsize=(10, 4))
    plt.plot(range(generation_counter), diversity_history, color='orange')
    plt.xlabel("Generation")
    plt.ylabel("Fitness Std Dev")
    plt.title("Fitness Diversity Over Generations")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fitness_diversity.png")
    plt.show()
    print("📊 Fitness diversity graph saved")

    # --- 🔹 Plot number of species ---
    plt.figure(figsize=(10, 4))
    plt.plot(range(generation_counter), species_count_history, color='purple')
    plt.xlabel("Generation")
    plt.ylabel("Number of Species")
    plt.title("Species Count Per Generation")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("species_count.png")
    plt.show()
    print("📊 Species count graph saved")
