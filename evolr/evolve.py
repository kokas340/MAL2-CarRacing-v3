import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
import neat
import gymnasium as gym
from agents.neat_agent import NeatAgent
from run_episode import run_agent


CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'neat_config.txt')
generation_counter = 0 
def eval_genome(genome, config):
    env = gym.make("CarRacing-v3", render_mode="none")  # No rendering for speed
    agent = NeatAgent(genome, config)
    score = run_agent(env, agent, render=False)
    env.close()
    return score

def eval_genomes(genomes, config):
    global generation_counter
    print(f"\n🚀 Evaluating Generation {generation_counter} with {len(genomes)} genomes")

    best_fitness = float("-inf")
    total_fitness = 0

    for genome_id, genome in genomes:
        fitness = eval_genome(genome, config)
        genome.fitness = fitness
        total_fitness += fitness
        if fitness > best_fitness:
            best_fitness = fitness

        print(f"  🧬 Genome {genome_id:>4} | Fitness: {fitness:7.2f}")

    avg_fitness = total_fitness / len(genomes)
    print(f"📈 Generation {generation_counter} Summary: Best: {best_fitness:.2f} | Avg: {avg_fitness:.2f}")
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
    population.add_reporter(neat.StatisticsReporter())

    winner = population.run(eval_genomes, n=100)

    print("\n🎉 Evolution complete!")
    print("Best genome:")
    print(winner)
    # Save the best genome to a file
    with open("best_genome.pkl", "wb") as f:
        pickle.dump(winner, f)
    print("✅ Best genome saved to best_genome.pkl")