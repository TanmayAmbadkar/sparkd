import gymnasium as gym
import torch
import numpy as np
from abstract_interpretation import domains, verification
import sys

class LunarLanderEnv(gym.Env):
    def __init__(self, state_processor=None, reduced_dim=None, safety=None):
        self.env = gym.make("LunarLander-v3", continuous=True, render_mode = "rgb_array")
        self.action_space = self.env.action_space
        self.original_observation_space = self.env.observation_space
        self.continuous = True
        
        self.observation_space = self.env.observation_space if state_processor is None else gym.spaces.Box(low=-1, high=1, shape=(reduced_dim,))
        self.state_processor = state_processor
        self.safety = safety

        self._max_episode_steps = 500
       
        self.step_counter = 0
        self.done = False  
        self.safe_polys = []
        self.polys = []
        self.transformed_polys = []
        self.transformed_safe_polys = []
        self.render_mode = "rgb_array"
        
        self.safety_constraints()
        self.unsafe_constraints()
        
        
    def safety_constraints(self):
        obs_space_lower = self.observation_space.low
        obs_space_upper = self.observation_space.high

        # Initialize lower and upper bounds as the observation space limits
        lower_bounds = np.copy(obs_space_lower)
        upper_bounds = np.copy(obs_space_upper)

        # Horizontal position constraint (x) - relaxed
        lower_bounds[0] = -1  # Increased from 0.75 to 1.0
        upper_bounds[0] = 1

        # Vertical position constraint (y) - relaxed
        
        lower_bounds[1] = -0.1
        upper_bounds[1] = 2

        # Horizontal velocity constraint (vx) - relaxed
        lower_bounds[2] = -1 # Increased from 0.5 to 0.75
        upper_bounds[2] = 1

        # Vertical velocity constraint (vy) - relaxed
        lower_bounds[3] = -1.5 # Increased from 0.5 to 0.75
        upper_bounds[3] = 0.5

        input_deeppoly = domains.DeepPoly(lower_bounds, upper_bounds)
    
        self.original_safe_polys = input_deeppoly.to_hyperplanes()
        self.safe_polys = input_deeppoly.to_hyperplanes()
        self.safety = input_deeppoly
        self.original_safety = input_deeppoly
 
        
    def unsafe_constraints(self):
        
        print(self.original_safety)
        unsafe_deeppolys = domains.recover_safe_region(domains.DeepPoly(self.observation_space.low, self.observation_space.high), [self.original_safety])        
        self.polys = []
        self.unsafe_domains = unsafe_deeppolys
        
        
        self.polys = unsafe_deeppolys.to_hyperplanes()

    def step(self, action):
        
        state, reward, done, truncation, info = self.env.step(action)
        self.done = done or self.step_counter >= self._max_episode_steps# Store the done flag

        original_state = np.copy(state)
        if self.state_processor is not None:
            # state = self.reduce_state(state)
            # state = torch.Tensor(state, dtype = torch.float64)
            with torch.no_grad():
                state = self.state_processor(state.reshape(1, -1))
            # state = state.numpy()
            state = state.reshape(-1,)
        # else:
            # state = self.reduce_state(state)
        self.step_counter+=1
        
        # if self.unsafe(state, simulated = False):
        #     self.done = True
        #     reward = -100
        
        
        return state, reward, self.done, truncation, {"state_original": original_state}

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)

        self.step_counter = 0
        self.done = False 
        original_state = np.copy(state)
        if self.state_processor is not None:
            # state = self.reduce_state(state)
            # state = torch.Tensor(state)
            with torch.no_grad():
                state = self.state_processor(state.reshape(1, -1))
            # state = state.numpy()
            state = state.reshape(-1,)
        # else:
            # state = self.reduce_state(state)
        return state, {"state_original": original_state}

    def render(self):
        return self.env.render()

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
        
        if simulated:
            for polys in self.safe_polys:
                
                A = polys[:,:-1]
                b = -polys[:,-1]
                return not np.all(A @ state.reshape(-1, 1) <= b.reshape(-1, 1))
        else:
            for polys in self.original_safe_polys:
                
                A = polys[:,:-1]
                b = -polys[:,-1]
                return not np.all(A @ state.reshape(-1, 1) <= b.reshape(-1, 1))
    


