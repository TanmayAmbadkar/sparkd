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
        
        self.observation_space = self.env.observation_space if state_processor is None else gym.spaces.Box(low=-1, high=1, shape=(reduced_dim,))
        self.state_processor = state_processor
        self.safety = safety

        self._max_episode_steps = 1600
       
        self.step_counter = 0
        self.done = False  
        self.safe_polys = []
        self.polys = []
        self.MAX = np.array(self.env.observation_space.high)
        self.MIN = np.array(self.env.observation_space.low)
        self.safety_constraints()
        self.unsafe_constraints()
        
    def safety_constraints(self):
        # Define the observation space bounds
        obs_space_lower = self.observation_space.low
        obs_space_upper = self.observation_space.high

        # Calculate the center of the observation space
        center = (obs_space_lower + obs_space_upper) / 2

        # Initialize the lower and upper bounds arrays
        lower_bounds = np.copy(obs_space_lower)
        upper_bounds = np.copy(obs_space_upper)

        # Modify the specific components based on the constraints
        # Angular Speed
        hull_ang_gen = 0.7
        vang_gen = 0.9
        vhor_gen = 0.8
        vver_gen = 0.8
        hipjoinang_gen = 1.1

        # Set the new bounds for each relevant component
        lower_bounds[0] = center[0] - hull_ang_gen
        upper_bounds[0] = center[0] + hull_ang_gen

        lower_bounds[1] = 0.25 - vang_gen
        upper_bounds[1] = 0.25 + vang_gen

        lower_bounds[2] = center[2] - vhor_gen
        upper_bounds[2] = center[2] + vhor_gen

        lower_bounds[3] = center[3] - vver_gen
        upper_bounds[3] = center[3] + vver_gen

        lower_bounds[4] = 0.15 - hipjoinang_gen
        upper_bounds[4] = 0.15 + hipjoinang_gen

        # Construct the polyhedra constraints (polys)
           
        lower_bounds = normalize_constraints(lower_bounds, a = self.MIN, b = self.MAX)
        upper_bounds = normalize_constraints(upper_bounds, a = self.MIN, b = self.MAX)
        input_deeppoly_domain = domains.DeepPoly(lower_bounds, upper_bounds)
        polys = input_deeppoly_domain.to_hyperplanes()

        # Set the safety constraints using the DeepPolyDomain and the polys
        self.safety = input_deeppoly_domain
        self.original_safety = input_deeppoly_domain
        self.safe_polys = [np.array(polys)]
        self.original_safe_polys = [np.array(polys)]
            
        
    def unsafe_constraints(self):
        
        obs_space_lower = self.observation_space.low
        obs_space_upper = self.observation_space.high
        unsafe_regions = []
        for polys in self.safe_polys:
            for i, poly in enumerate(polys):
                A = poly[:-1]
                b = -poly[-1]
                unsafe_regions.append(np.append(-A, b))
        
        for i in range(self.observation_space.shape[0]):
            A1 = np.zeros(self.observation_space.shape[0])
            A2 = np.zeros(self.observation_space.shape[0])
            A1[i] = 1
            A2[i] = -1
            unsafe_regions.append(np.append(A1, -obs_space_upper[i]))
            unsafe_regions.append(np.append(A2, obs_space_lower[i]))
        
        self.polys = [np.array(unsafe_regions)]

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
            original_state = normalize_constraints(original_state, self.MIN, self.MAX)
        else:
            state = normalize_constraints(state, self.MIN, self.MAX)
            
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
            original_state = normalize_constraints(original_state, self.MIN, self.MAX)
        else:
            state = normalize_constraints(state, self.MIN, self.MAX)
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
    


