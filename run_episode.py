import gymnasium as gym
from agents.random_agent import RandomAgent

def run_agent(env, agent, render=False, max_no_reward_steps=100):
    obs, _ = env.reset(seed=42)
    total_reward = 0
    last_reward = 0
    steps = 0
    no_reward_steps = 0

    done = False
    truncated = False

    while not done and not truncated:
        if render:
            env.render()

        action = agent.act(obs)
        obs, reward, done, truncated, _ = env.step(action)

        total_reward += reward
        steps += 1

        # Early stopping if no reward change
        if reward <= 0:
            no_reward_steps += 1
        else:
            no_reward_steps = 0

        if no_reward_steps >= max_no_reward_steps:
            break
        #no progress steps
        if reward == 0:
            no_progress_steps += 1
        else:
            no_progress_steps = 0
       

        if no_progress_steps > 60:
            break
    return total_reward


# --- Test run ---
if __name__ == "__main__":
    env = gym.make("CarRacing-v3", render_mode="none")
    agent = RandomAgent(env.action_space)

    score = run_agent(env, agent, render=True)
    print(f"✅ Episode finished with score: {score:.2f}")
    env.close()
