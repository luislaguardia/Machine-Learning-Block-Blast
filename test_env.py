# with commnent/ notes
# last updated: luis
# dec3mbwer 24, 2024

from block_blast_env import BlockBlastEnv

# Initialize the environment
env = BlockBlastEnv()

# Reset the environment
obs = env.reset()
done = False

# Play a random game
while not done:
    env.render()  # Show the grid
    action = env.action_space.sample()  # Take a random action
    obs, reward, done, info = env.step(action)  # Perform the action
    print(f"Reward: {reward}")
