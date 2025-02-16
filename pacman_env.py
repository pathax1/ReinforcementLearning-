import numpy as np
import gym
import pygame
import os
from gym import spaces

class PacmanEnv(gym.Env):
    def __init__(self):
        super(PacmanEnv, self).__init__()
        self.grid_size = (10, 10)
        self.action_space = spaces.Discrete(4)  # 4 actions: UP, DOWN, LEFT, RIGHT
        self.observation_space = spaces.Box(low=0, high=1, shape=(10, 10), dtype=np.uint8)

        self.window_size = 400
        self.cell_size = self.window_size // 10
        pygame.init()
        self.screen = pygame.Surface((self.window_size, self.window_size))

        self.reset()

    def reset(self):
        """Resets the environment to the initial state."""
        self.pacman_pos = [5, 5]
        self.food = [(2, 3), (7, 8), (4, 6)]
        self.ghosts = [(1, 1), (8, 8)]

        state = np.zeros((10, 10), dtype=int)
        state[self.pacman_pos[0], self.pacman_pos[1]] = 1  # Pac-Man
        for f in self.food:
            state[f[0], f[1]] = 2  # Food

        return state

    def step(self, action):
        """Updates the environment based on the agent's action."""
        if action == 0:  # UP
            self.pacman_pos[1] = max(0, self.pacman_pos[1] - 1)
        elif action == 1:  # DOWN
            self.pacman_pos[1] = min(9, self.pacman_pos[1] + 1)
        elif action == 2:  # LEFT
            self.pacman_pos[0] = max(0, self.pacman_pos[0] - 1)
        elif action == 3:  # RIGHT
            self.pacman_pos[0] = min(9, self.pacman_pos[0] + 1)

        reward = 0
        if tuple(self.pacman_pos) in self.food:
            reward += 10
            self.food.remove(tuple(self.pacman_pos))

        done = len(self.food) == 0

        state = np.zeros((10, 10), dtype=int)
        state[self.pacman_pos[0], self.pacman_pos[1]] = 1
        for f in self.food:
            state[f[0], f[1]] = 2

        return state, reward, done, {}

    def render(self):
        """Saves the game frame as an image for Flask."""
        self.screen.fill((0, 0, 0))

        for x in range(10):
            for y in range(10):
                pygame.draw.rect(self.screen, (40, 40, 40), (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size), 1)

        pygame.draw.circle(self.screen, (255, 255, 0),
                           (self.pacman_pos[0] * self.cell_size + self.cell_size // 2,
                            self.pacman_pos[1] * self.cell_size + self.cell_size // 2), self.cell_size // 3)

        for f in self.food:
            pygame.draw.circle(self.screen, (0, 255, 0), (f[0] * self.cell_size + self.cell_size // 2, f[1] * self.cell_size + self.cell_size // 2), self.cell_size // 5)

        img_path = "static/pacman_frame.png"
        pygame.image.save(self.screen, img_path)
        return img_path
