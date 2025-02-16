import numpy as np
from pacman_env import PacmanEnv

env = PacmanEnv()
q_table = np.zeros((10, 10, 4))  # 10x10 grid with 4 possible actions

learning_rate = 0.1
discount_factor = 0.9
epsilon = 1.0
epsilon_decay = 0.99
episodes = 500

for episode in range(episodes):
    state = env.reset()
    done = False

    while not done:
        pac_x, pac_y = env.pacman_pos
        action = env.action_space.sample() if np.random.uniform(0, 1) < epsilon else np.argmax(q_table[pac_x, pac_y])

        new_state, reward, done, _ = env.step(action)
        new_x, new_y = env.pacman_pos

        q_table[pac_x, pac_y, action] = (1 - learning_rate) * q_table[pac_x, pac_y, action] + \
            learning_rate * (reward + discount_factor * np.max(q_table[new_x, new_y]))

        state = new_state

    epsilon *= epsilon_decay

np.save("models/q_table.npy", q_table)
print("✅ Q-learning training completed!")
