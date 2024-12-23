# with comments
# last updated, luis
# december 24, 2024


import numpy as np
import gymnasium as gym
from gymnasium import spaces

class BlockBlastEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, grid_size=10):
        super(BlockBlastEnv, self).__init__()
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        self.blocks = [
            np.array([[1, 1], [1, 0]]),  # L-shape
            np.array([[1, 1], [0, 1]]),  # Reverse L-shape
            np.array([[1, 1, 1]]),       # Line
            np.array([[1, 1], [1, 1]])   # Square
        ]
        self.current_block = None

        # Define action and observation space
        self.action_space = spaces.Discrete(self.grid_size * self.grid_size)
        self.observation_space = spaces.Box(low=0, high=1, 
                                            shape=(grid_size, grid_size), 
                                            dtype=int)

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.current_block = self._get_random_block()
        return self._get_observation()

    def step(self, action):
        row, col = divmod(action, self.grid_size)
        block = self.current_block

        # Attempt to place block
        if self._can_place(block, row, col):
            self._place_block(block, row, col)
            reward = self._clear_rows_and_columns()
            self.current_block = self._get_random_block()
            done = not self._has_valid_moves()
            return self._get_observation(), reward, done, {}
        else:
            # Invalid move
            return self._get_observation(), -10, False, {}

    def _get_random_block(self):
        return self.blocks[np.random.randint(len(self.blocks))]

    def _get_observation(self):
        return self.grid

    def _can_place(self, block, row, col):
        block_height, block_width = block.shape
        if row + block_height > self.grid_size or col + block_width > self.grid_size:
            return False
        return np.all(self.grid[row:row+block_height, col:col+block_width] + block <= 1)

    def _place_block(self, block, row, col):
        block_height, block_width = block.shape
        self.grid[row:row+block_height, col:col+block_width] += block

    def _clear_rows_and_columns(self):
        cleared_rows = np.sum(np.all(self.grid == 1, axis=1))
        cleared_cols = np.sum(np.all(self.grid == 1, axis=0))
        reward = (cleared_rows + cleared_cols) * 10

        # Remove cleared rows and columns
        self.grid = np.delete(self.grid, np.where(np.all(self.grid == 1, axis=1)), axis=0)
        self.grid = np.delete(self.grid.T, np.where(np.all(self.grid == 1, axis=0)), axis=1).T

        # Refill grid to original size with zeros
        while self.grid.shape[0] < self.grid_size:
            self.grid = np.vstack((np.zeros((1, self.grid_size), dtype=int), self.grid))
        while self.grid.shape[1] < self.grid_size:
            self.grid = np.hstack((np.zeros((self.grid_size, 1), dtype=int), self.grid))

        return reward

    def _has_valid_moves(self):
        for block in self.blocks:
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    if self._can_place(block, row, col):
                        return True
        return False

    def render(self, mode='human'):
        print(self.grid)
