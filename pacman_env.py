import gym
import numpy as np
import pygame
from gym import spaces


class PacmanEnv(gym.Env):
    def __init__(self):
        super(PacmanEnv, self).__init__()
        self.grid_size = (10, 10)  # 10x10 grid
        self.action_space = spaces.Discrete(4)  # 4 moves: UP, DOWN, LEFT, RIGHT
        self.observation_space = spaces.Box(low=0, high=255, shape=(10, 10, 3), dtype=np.uint8)
        self.pacman_pos = [5, 5]  # Initial position
        self.food = [(2, 3), (7, 8)]  # Example food positions
        self.ghosts = [(1, 1), (8, 8)]

    def reset(self):
        self.pacman_pos = [5, 5]
        return np.zeros((10, 10, 3), dtype=np.uint8)

    def step(self, action):
        # Update pacman position based on action
        if action == 0:  # UP
            self.pacman_pos[1] -= 1
        elif action == 1:  # DOWN
            self.pacman_pos[1] += 1
        elif action == 2:  # LEFT
            self.pacman_pos[0] -= 1
        elif action == 3:  # RIGHT
            self.pacman_pos[0] += 1

        reward = 0
        if tuple(self.pacman_pos) in self.food:
            reward += 10  # Eating food

        done = False  # Add logic for winning/losing condition

        return np.zeros((10, 10, 3), dtype=np.uint8), reward, done, {}

    def render(self, mode="human"):
        # Use pygame for visualization
        pass
