import torch
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class CarPlatoonEnv(gym.Env):
    def __init__(self, state_processor=None, reduced_dim=None):
        super(CarPlatoonEnv, self).__init__()
        self._max_episode_steps = 500

        # Define the system matrices A and B
        self.A = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0.1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0.1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0.1],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        self.B = np.array([
            [0.1, 0, 0, 0],
            [0, 0, 0, 0],
            [0.1, -0.1, 0, 0],
            [0, 0, 0, 0],
            [0, 0.1, -0.1, 0],
            [0, 0, 0, 0],
            [0, 0, 0.1, -0.1]
        ])

        # Define the observation and action spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(4,), dtype=np.float32)

        # State processor for dimensionality reduction
        self.state_processor = state_processor
        self.reduced_dim = reduced_dim
        self.done = False
        self.state = None
        self.reset()

    def reset(self):
        self.state = np.random.uniform(low=[19.9, 0.9, -0.1, 0.9, -0.1, 0.9, -0.1],
                                       high=[20.1, 1.1, 0.1, 1.1, 0.1, 1.1, 0.1])
        self.done = False
        print(f"Initial state: {self.state}")  # Debug print
        return self._get_observation()

    def step(self, action):
        u = np.clip(action, self.action_space.low, self.action_space.high)
        print(f"Action taken: {u}")  # Debug print
        self.state = np.dot(self.A, self.state) + np.dot(self.B, u)
        print(f"New state: {self.state}")  # Debug print

        reward = self._reward(self.state, u)
        self.done = self._done(self.state)
        print(f"Done: {self.done}")  # Debug print

        return self._get_observation(), reward, self.done, {}

    def _reward(self, state, action):
        # Define the reward function
        Q = np.eye(7)
        R = np.eye(4) * 0.0005
        return - (state.T @ Q @ state + action.T @ R @ action).item()

    def _done(self, state):
        # Define the termination condition with less strict bounds
        x_min = np.array([17, 0.3, -0.5, 0.3, -1.5, 0.3, -1.5])
        x_max = np.array([23, 1.7, 0.5, 1.7, 1.5, 1.7, 1.5])
        return np.any(state < x_min) or np.any(state > x_max)

    def _get_observation(self):
        if self.state_processor is not None:
            state = torch.Tensor(self.state)
            with torch.no_grad():
                state = self.state_processor(state.reshape(1, -1))
            state = state.numpy()
            return state
        else:
            return self.state if self.reduced_dim is None else self.state[:self.reduced_dim]

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            self.action_space.seed(seed)
            self.observation_space.seed(seed)

    def predict_done(self, state: np.ndarray) -> bool:
        return self.done

    def unsafe(self, state: np.ndarray) -> bool:
        # Define the unsafe region with less strict bounds
        x_min = np.array([17, 0.3, -0.5, 0.3, -1.5, 0.3, -1.5])
        x_max = np.array([23, 1.7, 0.5, 1.7, 1.5, 1.7, 1.5])
        return np.any(state < x_min) or np.any(state > x_max)
