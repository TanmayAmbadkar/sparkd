import gymnasium as gym
import torch
import numpy as np
from constraints import safety, verification
import sys
from gymnasium.wrappers import NormalizeObservation
class AntEnv(gym.Env):
    def __init__(self, state_processor=None, reduced_dim=None, safety=None):
        self.env = gym.make("Ant-v5", render_mode="rgb_array")
        self.action_space = self.env.action_space
        
        self.observation_space = self.env.observation_space if state_processor is None else gym.spaces.Box(low=-1, high=1, shape=(reduced_dim,))
        self.state_processor = state_processor
        self.safety = safety
        print(self.action_space.low, self.action_space.high)

        self._max_episode_steps = 1000
       
        self.step_counter = 0
        self.done = False  
        self.safe_polys = []
        self.polys = []
        
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


        # Initialize the lower and upper bounds arrays
        lower_bounds = np.copy(obs_space_lower)
        upper_bounds = np.copy(obs_space_upper)
        lower_bounds = np.nan_to_num(lower_bounds, nan=-9999, posinf=33333333, neginf=-33333333)
        upper_bounds = np.nan_to_num(upper_bounds, nan=-9999, posinf=33333333, neginf=-33333333)

        # lower_bounds[:12] = [ -4.12, -18.4, 9.80, -0.63, -0.18, -0.1,     -0.1,     -0.1,    -3,    -0.5, -0.51,   -0.1,  ]
        # upper_bounds[:12] =  [ 4.01, 18.39,  9.82,  0.72,  0.15,  0.1,    0.1,    0.1,   3,    0.5,   0.51,  0.1,  ]
        
        # for i in range(12, 28):
        #     lower_bounds[i] = 0
        #     upper_bounds[i] = 
        lower_bounds[13] = -2.3475
        upper_bounds[13] = 2.3475
        
        # lower_bounds = normalize_constraints(lower_bounds, a = self.MIN, b = self.MAX, target_range=(-1, 1))
        # upper_bounds = normalize_constraints(upper_bounds, a = self.MIN, b = self.MAX, target_range=(-1, 1))
        
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

        original_state = np.copy(state)
        
        self.step_counter+=1
        
        return state, reward, self.done, truncation, {"state_original": original_state}

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)

        self.step_counter = 0
        self.done = False 
        original_state = np.copy(state)
       
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
        
        is_health = 2.3475 >= state[13] >= -2.3475
        return not is_health

