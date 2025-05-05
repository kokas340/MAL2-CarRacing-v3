import gymnasium as gym
from agents.random_agent import RandomAgent

def run_agent(env, agent, render=False, max_no_reward_steps=50):
    obs, _ = env.reset(seed=42)
    total_reward = 0
    steps = 0
    no_reward_steps = 0
    no_progress_steps = 0

    done = False
    truncated = False

    while not done and not truncated:
        if render:
            env.render()

        action = agent.act(obs)
        obs, reward, done, truncated, _ = env.step(action)

        total_reward += reward
        steps += 1

        # Early stopping if no reward progress
        if reward <= 0:
            no_reward_steps += 1
        else:
            no_reward_steps = 0

        if reward == 0:
            no_progress_steps += 1
        else:
            no_progress_steps = 0

        if no_reward_steps >= max_no_reward_steps:
            break

    return total_reward, steps, no_reward_steps, no_progress_steps


