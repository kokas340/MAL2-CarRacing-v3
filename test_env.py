import pygame
from envs.robot_vacuum_env import RobotVacuumEnv
import time

env = RobotVacuumEnv(grid_size=6, dirt_count=8)
obs, _ = env.reset()

done = False
step_count = 0

# 💡 FIRST call render() to initialize pygame's display before using event.get()
env.render()

while not done and step_count < 100:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    action = env.action_space.sample()
    obs, reward, done, _, _ = env.step(action)

    print(f"Step {step_count} | Action: {action} | Reward: {reward}")
    env.render()
    step_count += 1

env.close()
