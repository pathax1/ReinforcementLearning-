import streamlit as st
import numpy as np
from pacman_env import PacmanEnv
from q_learning import q_table

env = PacmanEnv()

st.title("Pac-Man RL Agent")

st.sidebar.write("Choose Model:")
model = st.sidebar.selectbox("Model", ["Q-learning", "Monte Carlo"])

if st.button("Run Simulation"):
    state = env.reset()
    done = False
    while not done:
        action = np.argmax(q_table[state[0], state[1]]) if model == "Q-learning" else env.action_space.sample()
        state, _, done, _ = env.step(action)
        env.render()
