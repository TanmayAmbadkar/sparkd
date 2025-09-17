import gymnasium as gym
import torch
import numpy as np
from constraints import safety, verification
import sys

class SimulatedEnv(gym.Env):
    def __init__(self, real_env, simulated_env, render_mode = None):
        self.env = real_env
        self.simulated_env = simulated_env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.continuous = True
        self._max_episode_steps = self.env._max_episode_steps
       
        self.step_counter = 0
        self.render_mode = render_mode
        

    def step(self, action):
        
        next_state, reward = self.simulated_env(self.state, action)

        done = self.env.unsafe(next_state, simulated = True)
        trunc = self.step_counter == self._max_episode_steps
        self.state = next_state

        if done:
            reward+=-1000

        return next_state, reward, done, trunc, {}

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        self.state = state

        self.step_counter = 0

        return state, {}


