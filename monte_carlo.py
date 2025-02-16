import numpy as np
from pacman_env import PacmanEnv

env = PacmanEnv()
returns = {}  # Dictionary to store returns for state-action pairs

for episode in range(5):
    episode_data = []
    state = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # Random action
        new_state, reward, done, _ = env.step(action)
        episode_data.append((state, action, reward))
        state = new_state

    total_reward = 0
    for step in reversed(episode_data):
        state, action, reward = step
        total_reward += reward
        returns[(state, action)] = returns.get((state, action), []) + [total_reward]
