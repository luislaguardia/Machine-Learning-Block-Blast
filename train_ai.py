# with comments
# last updated: ;luis
# december 24,2-024

from block_blast_env import BlockBlastEnv
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

# Wrap the environment for reinforcement learning
env = make_vec_env(lambda: BlockBlastEnv(), n_envs=1)

# Create the DQN model
model = DQN("MlpPolicy", env, verbose=1)

# Train the model
print("Training the AI...")
model.learn(total_timesteps=10000)

# Save the trained model
model.save("block_blast_dqn")
print("Model saved as 'block_blast_dqn'")
