import numpy as np
from pacman_env import PacmanEnv

env = PacmanEnv()
mc_policy = np.zeros((10, 10))

episodes = 500
returns = {}

for episode in range(episodes):
    episode_data = []
    state = env.reset()
    done = False

    while not done:
        pac_x, pac_y = env.pacman_pos
        action = env.action_space.sample()

        new_state, reward, done, _ = env.step(action)
        new_x, new_y = env.pacman_pos

        episode_data.append(((pac_x, pac_y), action, reward))
        state = new_state

    total_reward = 0
    for step in reversed(episode_data):
        (pac_x, pac_y), action, reward = step
        total_reward += reward
        if (pac_x, pac_y, action) not in returns:
            returns[(pac_x, pac_y, action)] = []
        returns[(pac_x, pac_y, action)].append(total_reward)

for key in returns:
    pac_x, pac_y, action = key
    mc_policy[pac_x, pac_y] = np.mean(returns[key])

np.save("models/mc_policy.npy", mc_policy)
print("✅ Monte Carlo training completed!")
