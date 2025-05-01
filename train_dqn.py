from envs.robot_vacuum_env import RobotVacuumEnv
from agents.dqn_agent import DQNAgent
import numpy as np

env = RobotVacuumEnv(grid_size=5, dirt_count=5)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = DQNAgent(state_size, action_size)
episodes = 500

reward_history = []

for ep in range(episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False
    steps = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.train()
        state = next_state
        total_reward += reward
        steps += 1

    reward_history.append(total_reward)
    print(f"[Episode {ep + 1}/{episodes}] Reward: {total_reward:.2f} | Steps: {steps} | Epsilon: {agent.epsilon:.3f}")

# Save model after training
agent.model.save("robot_dqn_model.h5")
print("✅ Model saved as robot_dqn_model.h5")

env.close()
