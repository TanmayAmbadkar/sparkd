import safety_gymnasium as gym
import gymnasium
import torch
import numpy as np
from abstract_interpretation import domains, verification
import sys
from benchmarks.utils import *

class SafetyPointGoalEnv(gymnasium.Env):
    def __init__(self, state_processor=None, reduced_dim=None, safety=None):
        self.env = gym.make("SafetyPointGoal1-v0")
        self.action_space = self.env.action_space
        
        self.observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=self.env.observation_space.shape) if state_processor is None else gymnasium.spaces.Box(low=-1, high=1, shape=(reduced_dim,))
        self.state_processor = state_processor
        self.safety = safety

        self._max_episode_steps = 1000
       
        self.step_counter = 0
        self.done = False  
        self.safe_polys = []
        self.polys = []
        
        self.MIN = np.array([-5, -19, 9.8, -0.8, -0.2, -0.1, -0.1, -0.1, -3, -0.5,
                -0.52, -0.1, 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

        self.MAX = np.concatenate((np.array([5, 19, 9.82, 0.8, 0.2, 0.1, 0.1, 0.1, 3., 0.5,
                0.52, 0.1, ]), np.ones(48)))

        self.safety_constraints()
        self.unsafe_constraints()
        
        # print(self.unsafe(np.array([ 0.41278508,  0.11044428,  0.03596416, -0.0501044,  -0.520235,   -0.7669368,
        #         0.55146146, -1.,          0.,         -0.3183163,  -1.0000002,   0.109326,
        #         0.9999997,   0.,          0.46180838,  0.4670529,   0.48339868,  0.51286566,
        #         0.55954015,  0.63115406,  0.7429231,   0.92812556,  1.,          1.,        ])))
        # sys.exit()
        
        
        
    def safety_constraints(self):
        # Define the observation space bounds
        obs_space_lower = self.observation_space.low
        obs_space_upper = self.observation_space.high

        # Calculate the center of the observation space
        center = (obs_space_lower + obs_space_upper) / 2

        # Initialize the lower and upper bounds arrays
        lower_bounds = np.copy(obs_space_lower)
        upper_bounds = np.copy(obs_space_upper)

        lower_bounds[:12] = [ -4.12, -18.4, 9.80, -0.63, -0.18, -0.1,     -0.1,     -0.1,    -3,    -0.5, -0.51,   -0.1,  ]
        upper_bounds[:12] =  [ 4.01, 18.39,  9.82,  0.72,  0.15,  0.1,    0.1,    0.1,   3,    0.5,   0.51,  0.1,  ]
        
        for i in range(12, 28):
            lower_bounds[i] = 0
            upper_bounds[i] = 1
            
        for i in range(28, 60):
            lower_bounds[i] = 0
            upper_bounds[i] = 0.8
            
        lower_bounds = normalize_constraints(lower_bounds, a = self.MIN, b = self.MAX, target_range=(-1, 1))
        upper_bounds = normalize_constraints(upper_bounds, a = self.MIN, b = self.MAX, target_range=(-1, 1))
        
        input_deeppoly_domain = domains.DeepPoly(lower_bounds, upper_bounds)
        polys = input_deeppoly_domain.to_hyperplanes()
        
        # Set the safety constraints using the DeepPolyDomain and the polys
        self.safety = input_deeppoly_domain
        self.original_safety = input_deeppoly_domain
        self.safe_polys = [np.array(polys)]
        self.original_safe_polys = [np.array(polys)]
        print(self.original_safety)
        # print(self.observation_space)
        
    def unsafe_constraints(self):
        
        unsafe_deeppolys = domains.recover_safe_region(domains.DeepPoly(self.observation_space.low, self.observation_space.high), [self.original_safety])        
        self.polys = []
        self.unsafe_domains = unsafe_deeppolys
        
        
        for poly in unsafe_deeppolys:
            self.polys.append(np.array(poly.to_hyperplanes()))
            
        
    def step(self, action):
        
        state, reward, cost, done, truncation, info = self.env.step(action)
        self.done = done or self.step_counter >= self._max_episode_steps# Store the done flag

        original_state = np.copy(state)
        if self.state_processor is not None:
            # state = self.reduce_state(state)
            # state = torch.Tensor(state, dtype = torch.float64)
            with torch.no_grad():
                state = self.state_processor(state.reshape(1, -1))
            # state = state.numpy()
            state = state.reshape(-1,)
            original_state = normalize_constraints(original_state, self.MIN, self.MAX, target_range=(-1, 1))
        else:
            state = normalize_constraints(state, self.MIN, self.MAX, target_range=(-1, 1))
            
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
            original_state = normalize_constraints(original_state, self.MIN, self.MAX, target_range=(-1, 1))
        else:
            state = normalize_constraints(state, self.MIN, self.MAX, target_range=(-1, 1))
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
            
            truth = []
            for polys in self.safe_polys:
                
                A = polys[:,:-1]
                b = -polys[:,-1]
                truth.append(not np.all(A @ state.reshape(-1, 1) <= b.reshape(-1, 1)))
            return all(truth)
        else:
            truth = []
            for polys in self.original_safe_polys:
                
                A = polys[:,:-1]
                b = -polys[:,-1]
                # print(A @ state.reshape(-1, 1) <= b.reshape(-1, 1))
                truth.append(not np.all(A @ state.reshape(-1, 1) <= b.reshape(-1, 1)))
            
            return all(truth)
    


