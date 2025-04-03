import gymnasium as gym
import torch
import numpy as np
import sys

class LunarLanderEnv(gym.Env):
    def __init__(self, state_processor=None, reduced_dim=None, safety=None):
        self.env = gym.make("LunarLander-v3", continuous=True, render_mode = "rgb_array")
        self.action_space = self.env.action_space
        
        self.observation_space = self.env.observation_space if state_processor is None else gym.spaces.Box(low=-1, high=1, shape=(reduced_dim,))
        
        self._max_episode_steps = 500
       
        self.step_counter = 0
        self.done = False  
        self.safe_polys = []
        self.polys = []
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
        lower_bounds[2] = -1.5 # Increased from 0.5 to 0.75
        upper_bounds[2] = 1.5

        # Vertical velocity constraint (vy) - relaxed
        lower_bounds[3] = -1.5  # Increased from 0.5 to 0.75
        upper_bounds[3] = 1.5

        polys = []
        for i in range(self.observation_space.shape[0]):
            A1 = np.zeros(self.observation_space.shape[0])
            A2 = np.zeros(self.observation_space.shape[0])

            # Upper bound constraint: A[i] * x[i] <= u_i
            A1[i] = 1
            polys.append(np.append(A1, -upper_bounds[i]))

            # Lower bound constraint: A[i] * x[i] >= l_i, or -A[i] * x[i] <= -l_i
            A2[i] = -1
            polys.append(np.append(A2, lower_bounds[i]))

        self.safe_polys = [np.array(polys)]
        # Set the safety constraints using t
        
        
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

        self.step_counter+=1
        
        
        
        return state, reward, self.done, {}

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)

        self.step_counter = 0
        self.done = False 
        return state

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

    def unsafe(self, state: np.ndarray) -> bool:
        
        for polys in self.safe_polys:
                
            A = polys[:,:-1]
            b = -polys[:,-1]
            return not np.all(A @ state.reshape(-1, 1) <= b.reshape(-1, 1))