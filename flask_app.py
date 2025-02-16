from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from pacman_env import PacmanEnv

app = Flask(__name__, template_folder="templates")

env = PacmanEnv()

# Ensure models exist before loading
if not os.path.exists("models/q_table.npy") or not os.path.exists("models/mc_policy.npy"):
    print("⚠️ ERROR: Model files not found! Please run `train_q_learning.py` and `train_monte_carlo.py` first.")
    exit(1)

q_table = np.load("models/q_table.npy")
mc_policy = np.load("models/mc_policy.npy")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/run_game', methods=['POST'])
def run_game():
    try:
        data = request.get_json()
        model_type = data.get("model")  # Get the selected model from the request

        if model_type not in ["Q-learning", "Monte Carlo"]:
            return jsonify({"error": "Invalid model type"}), 400

        state = env.reset()
        done = False
        frames = []

        while not done:
            pac_x, pac_y = env.pacman_pos  # Extract valid indices

            if model_type == "Q-learning":
                action = np.argmax(q_table[pac_x, pac_y])
            else:  # Monte Carlo
                action = mc_policy[pac_x, pac_y]

            state, _, done, _ = env.step(int(action))
            frame_path = env.render()
            frames.append(frame_path)

        return jsonify({"frames": frames})  # Return the list of image paths

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
