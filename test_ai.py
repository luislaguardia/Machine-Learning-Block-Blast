# WIUTH COMMEHTNS
# last updatedL luius
# devcember 24, 2024

from block_blast_env import BlockBlastEnv
from stable_baselines3 import DQN

# Load the environment and trained model
env = BlockBlastEnv()
model = DQN.load("block_blast_dqn")

# Test the AI
obs = env.reset()
done = False

print("Testing the AI...")
while not done:
    env.render()
    action, _states = model.predict(obs)  # AI selects an action
    obs, reward, done, info = env.step(action)  # Perform the action
    print(f"Reward: {reward}")
