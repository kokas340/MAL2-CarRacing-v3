import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class RobotVacuumEnv(gym.Env):
    def __init__(self, grid_size=5, dirt_count=5, max_steps=200):
        self.grid_size = grid_size
        self.dirt_count = dirt_count
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(self.grid_size * self.grid_size,), dtype=np.int32
        )


        self.reset()

    def reset(self, seed=None, options=None):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.steps = 0

        # Place dirt
        dirt_positions = np.random.choice(self.grid_size**2, self.dirt_count, replace=False)
        for pos in dirt_positions:
            x, y = divmod(pos, self.grid_size)
            self.grid[x, y] = 1

        # Place agent
        while True:
            x = np.random.randint(self.grid_size)
            y = np.random.randint(self.grid_size)
            if self.grid[x, y] == 0:
                self.agent_pos = [x, y]
                break

        return self._get_obs(), {}

    def _get_obs(self):
        obs = self.grid.copy()
        x, y = self.agent_pos
        obs[x, y] = 2  # Mark robot position
        return obs.flatten()  # Flatten for MLP input


    def step(self, action):
        x, y = self.agent_pos
        reward = -1  # Default step penalty
        done = False

        # Movement
        if action == 0 and x > 0:
            x -= 1
        elif action == 1 and x < self.grid_size - 1:
            x += 1
        elif action == 2 and y > 0:
            y -= 1
        elif action == 3 and y < self.grid_size - 1:
            y += 1
        elif action == 4:  # Clean
            if self.grid[x, y] == 1:
                self.grid[x, y] = 0
                reward = 10

        self.agent_pos = [x, y]
        self.steps += 1

        # DEBUG (optional): Print small logs for tracking
        if self.steps % 50 == 0:
            print(f"Step {self.steps} | Position: {self.agent_pos} | Dirt Left: {np.sum(self.grid == 1)}")

        # Check termination condition
        if np.sum(self.grid == 1) == 0:
            print(f"[INFO] All dirt cleaned at step {self.steps}")
            done = True
        elif self.steps >= self.max_steps:
            print(f"[INFO] Max steps reached: {self.steps}")
            done = True

        return self._get_obs(), reward, done, False, {}

    
    def render(self, mode="human"):
        # Lazy init
        if not hasattr(self, 'screen'):
            pygame.init()
            self.cell_size = 80  # 🔥 Make cells larger
            self.window_size = self.grid_size * self.cell_size
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("🧹 Robot Vacuum RL")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 20)

        self.screen.fill((240, 240, 240))  # Background = light gray

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x = j * self.cell_size
                y = i * self.cell_size

                # Floor tile
                pygame.draw.rect(self.screen, (200, 200, 200), (x, y, self.cell_size, self.cell_size))

                # Grid lines
                pygame.draw.rect(self.screen, (160, 160, 160), (x, y, self.cell_size, self.cell_size), 2)

                # Dirt
                if self.grid[i, j] == 1:
                    pygame.draw.circle(
                        self.screen,
                        (139, 69, 19),  # Dirt brown
                        (x + self.cell_size // 2, y + self.cell_size // 2),
                        self.cell_size // 5,
                    )

        # Draw the robot
        rx, ry = self.agent_pos
        robot_x = ry * self.cell_size
        robot_y = rx * self.cell_size

        pygame.draw.circle(
            self.screen,
            (30, 144, 255),  # Dodger blue
            (robot_x + self.cell_size // 2, robot_y + self.cell_size // 2),
            self.cell_size // 3,
        )

        # Optional text info
        step_text = self.font.render(f"Steps: {self.steps}", True, (0, 0, 0))
        self.screen.blit(step_text, (10, 10))

        pygame.display.flip()
        self.clock.tick(10)  # Limit FPS

    def close(self):
        if hasattr(self, 'screen'):
            pygame.quit()