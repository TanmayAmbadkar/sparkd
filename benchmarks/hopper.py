import gymnasium as gym
import torch
import numpy as np
from constraints import safety
import sys

class HopperEnv(gym.Env):
    def __init__(self, state_processor=None, reduced_dim=None, safety=None):
        self.env = gym.make("Hopper-v4")
        self.action_space = self.env.action_space
        
        self.observation_space = self.env.observation_space if state_processor is None else gym.spaces.Box(low=-1, high=1, shape=(reduced_dim,))
        self.state_processor = state_processor
        self.safety = safety

        self._max_episode_steps = 1000
       
        self.step_counter = 0
        self.done = False  
        self.safe_polys = []
        self.polys = []
        
        self.safety_constraints()
        self.unsafe_constraints()
        
    def safety_constraints(self):
        # Define the observation space bounds
        obs_space_lower = self.observation_space.low
        obs_space_upper = self.observation_space.high


        # Initialize the lower and upper bounds arrays
        lower_bounds = np.copy(obs_space_lower)
        upper_bounds = np.copy(obs_space_upper)
        lower_bounds = np.nan_to_num(lower_bounds, nan=-9999, posinf=33333333, neginf=-33333333)
        upper_bounds = np.nan_to_num(upper_bounds, nan=-9999, posinf=33333333, neginf=-33333333)
        lower_bounds[5] = -0.37315
        upper_bounds[5] = 0.37315
        input_Box_domain = safety.Box(lower_bounds, upper_bounds)
        polys = input_Box_domain.to_hyperplanes(self.env.observation_space)
        
        # Set the safety constraints using the BoxDomain and the polys
        self.safety = input_Box_domain
        self.original_safety = input_Box_domain
        self.safe_polys = polys
        self.original_safe_polys = polys
        print(self.original_safety)
        # print(self.observation_space)
        
    def unsafe_constraints(self):
        
        self.polys = self.safety.invert_polytope(self.env.observation_space)
        print(len(self.polys))
            

    def step(self, action):
        
        state, reward, done, truncation, info = self.env.step(action)
        self.done = done or self.step_counter >= self._max_episode_steps# Store the done flag

        self.step_counter+=1
        
        return state, reward, self.done, truncation, {}

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)

        self.step_counter = 0
        self.done = False 
       
        return state, {}

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            self.env.action_space.seed(seed)
            self.env.observation_space.seed(seed)

    def predict_done(self, state: np.ndarray) -> bool:
        return self.done

    def unsafe(self, state: np.ndarray, simulated:bool = False) -> bool:
       
       is_healthy =  -0.37315 <= state[5] <= 0.37315
       
       return not is_healthy