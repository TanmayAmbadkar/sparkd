import gymnasium as gym
import numpy as np

class OscillatorEnv(gym.Env):
    def __init__(self, max_episode_steps=500):
        self.ds = 18
        self.us = 2
        self.h = 0.01

        self.s_min = np.array([0.2, -0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.s_max = np.array([0.3, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        self.u_min = np.array([-50., -50])
        self.u_max = np.array([50., 50])

        self.Q = np.zeros((self.ds, self.ds), float)
        self.R = np.zeros((self.us, self.us), float)
        np.fill_diagonal(self.Q, 1)
        np.fill_diagonal(self.R, 1)

        self.action_space = gym.spaces.Box(low=self.u_min, high=self.u_max, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=self.s_min, high=self.s_max, dtype=np.float32)

        self.state = None
        self.step_counter = 0
        self._max_episode_steps = max_episode_steps

    def f(self, x, u):
        delta = np.zeros((self.ds, 1), float)
        delta[0, 0] = -2 * x[0, 0] + u[0, 0]
        delta[1, 0] = -x[1, 0] + u[1, 0]
        for i in range(2, self.ds):
            delta[i, 0] = 5 * x[i-2, 0] - 5 * x[i, 0]
        return delta

    def step(self, action):
        u = action.reshape(self.us, 1)
        delta = self.f(self.state, u)
        self.state += self.h * delta
        reward = self._reward(self.state, u)
        done = self.unsafe(self.state) or self.step_counter >= self._max_episode_steps
        self.step_counter += 1
        return self.state.flatten(), reward, done, {}

    def reset(self):
        self.state = np.random.uniform(self.s_min, self.s_max).reshape(self.ds, 1)
        self.step_counter = 0
        return self.state.flatten()

    def _reward(self, state, action):
        reward = - (state.T @ self.Q @ state + action.T @ self.R @ action)[0, 0]
        if self.unsafe(state):
            reward -= 100
        return reward

    def unsafe(self, state):
        return state[17, 0] >= 0.05

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            self.action_space.seed(seed)
            self.observation_space.seed(seed)
