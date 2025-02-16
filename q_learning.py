import numpy as np
import random
from pacman_env import PacmanEnv

env = PacmanEnv()
q_table = np.zeros((10, 10, 4))  # Grid world (10x10) with 4 actions

learning_rate = 0.1
discount_factor = 0.9
epsilon = 1.0  # Start with full exploration
epsilon_decay = 0.99

for episode in range(5):
    state = env.reset()
    done = False
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state[0], state[1]])  # Exploit

        new_state, reward, done, _ = env.step(action)
        q_table[state[0], state[1], action] = (1 - learning_rate) * q_table[
            state[0], state[1], action] + learning_rate * (
                                                      reward + discount_factor * np.max(
                                                  q_table[new_state[0], new_state[1]])
                                              )

        state = new_state

    epsilon *= epsilon_decay
