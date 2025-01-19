import gymnasium as gym
import torch
import numpy as np
from abstract_interpretation import domains, verification
import sys
from benchmarks.utils import *

class BipedalWalkerEnv(gym.Env):
    def __init__(self, state_processor=None, reduced_dim=None, safety=None):
        self.env = gym.make("BipedalWalker-v3")
        self.action_space = self.env.action_space
        
        self.original_observation_space = self.env.observation_space
        self.observation_space = self.env.observation_space if state_processor is None else gym.spaces.Box(low=-1, high=1, shape=(reduced_dim,))
        # self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(self.env.observation_space.shape[0],)) if state_processor is None else gym.spaces.Box(low=-1, high=1, shape=(reduced_dim,))
        
        self.state_processor = state_processor
        self.safety = safety

        self._max_episode_steps = 1600
       
        self.step_counter = 0
        self.done = False  
        self.safe_polys = []
        self.polys = []
        self.safety_constraints()
        self.unsafe_constraints()
        
    def safety_constraints(self):
        # Define the observation space bounds
        obs_space_lower = self.original_observation_space.low
        obs_space_upper = self.original_observation_space.high

        # Calculate the center of the observation space
        center = (obs_space_lower + obs_space_upper) / 2

        # Initialize the lower and upper bounds arrays
        lower_bounds = np.copy(obs_space_lower)
        upper_bounds = np.copy(obs_space_upper)

        # Modify the specific components based on the constraints
        # Angular Speed
        # Set the new bounds for each relevant component
        lower_bounds[0] = -1
        upper_bounds[0] = 1

        lower_bounds[1] = -1
        upper_bounds[1] = 1

        lower_bounds[2] = -1
        upper_bounds[2] = 1

        lower_bounds[3] = -1
        upper_bounds[3] = 1

        lower_bounds[4] = -1.5
        upper_bounds[4] = 1.5
        
        lower_bounds[5] = -2
        upper_bounds[5] = 2
        
        lower_bounds[6] = -2
        upper_bounds[6] = 2
        
        lower_bounds[7] = -3
        upper_bounds[7] = 3
        
        lower_bounds[8] = 0
        upper_bounds[8] = 2
        
        lower_bounds[9] = -2
        upper_bounds[9] = 2
        
        lower_bounds[10] = -2
        upper_bounds[10] = 2
        
        lower_bounds[11] = -1
        upper_bounds[11] = 1
        
        lower_bounds[12] = -2
        upper_bounds[12] = 2
        
        lower_bounds[13] = -0.01
        upper_bounds[13] = 1.01


        # Construct the polyhedra constraints (polys)
        
        print(lower_bounds, upper_bounds)
           
        input_deeppoly_domain = domains.DeepPoly(lower_bounds, upper_bounds)
        polys = input_deeppoly_domain.to_hyperplanes()

        # Set the safety constraints using the DeepPolyDomain and the polys
        self.safety = input_deeppoly_domain
        self.original_safety = input_deeppoly_domain
        self.safe_polys = [np.array(polys)]
        self.original_safe_polys = [np.array(polys)]
      
    def unsafe_constraints(self):
        
        unsafe_deeppolys = domains.recover_safe_region(domains.DeepPoly(self.observation_space.low, self.observation_space.high), [self.original_safety])        
        self.polys = []
        self.unsafe_domains = unsafe_deeppolys
        
        
        for poly in unsafe_deeppolys:
            self.polys.append(np.array(poly.to_hyperplanes()))

    def step(self, action):
        
        state, reward, done, truncation, info = self.env.step(action)
        self.done = done or self.step_counter >= self._max_episode_steps# Store the done flag

        original_state = np.copy(state)
        
        if self.state_processor is not None:
            # state = self.reduce_state(state)
            # state = torch.Tensor(state, dtype = torch.float64)
            
            with torch.no_grad():
                state = self.state_processor(original_state.reshape(1, -1))
            # state = state.numpy()
            state = state.reshape(-1,)
        
        

            
        self.step_counter+=1

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
            # original_state = normalize_constraints(original_state, self.MIN, self.MAX)
        return state, {"state_original": original_state}

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
        
        if simulated:
            for polys in self.safe_polys:
                
                A = polys[:,:-1]
                b = -polys[:,-1]
                return not np.all(A @ state.reshape(-1, 1) <= b.reshape(-1, 1))
        else:
            for polys in self.original_safe_polys:
                
                A = polys[:,:-1]
                b = -polys[:,-1]
                # print(A @ state.reshape(-1, 1) <= b.reshape(-1, 1))
                return not np.all(A @ state.reshape(-1, 1) <= b.reshape(-1, 1))
    


