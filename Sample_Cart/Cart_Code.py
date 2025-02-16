import gym  # Import the Gym library, which provides various RL environments


def cartpole():
    """
    Function to interact with the CartPole-v1 environment.
    It initializes the environment, takes random actions for 50 steps, and prints observations.
    """
    # Initialize the CartPole environment with 'rgb_array' render mode
    environment = gym.make('CartPole-v1', render_mode="rgb_array")

    # Reset the environment to its initial state
    environment.reset()

    # Loop for 50 steps
    for step in range(50):
        environment.render()  # Render the environment (use 'human' mode if visual output is needed)

        # Select a random action (0: Move left, 1: Move right)
        action = environment.action_space.sample()

        # Apply the action to the environment
        result = environment.step(action)

        # Handle Gym returning 4 or 5 values
        if len(result) == 4:
            observation, reward, done, info = results
        elif len(result) == 5:
            observation, reward, done, _, info = result  # Ignore the unused fifth value
        else:
            raise ValueError(f"Expected 4 or 5 values from environment.step(action), got {len(result)}")

        # Print step information
        print(f"Step {step + 1}:")
        print(f"Action: {action}")
        print(f"Observation: {observation}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Info: {info}\n")

        # If the episode ends, reset the environment
        if done:
            environment.reset()

    # Close the environment after execution
    environment.close()


# Run the function
if __name__ == "__main__":
    cartpole()
