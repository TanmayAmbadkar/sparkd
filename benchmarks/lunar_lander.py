import gymnasium as gym
import torch
import numpy as np
from abstract_interpretation import domains, verification
import sys

class LunarLanderEnv(gym.Env):
    def __init__(self, state_processor=None, reduced_dim=None, safety=None):
        self.env = gym.make("LunarLander-v3", continuous=True)
        self.action_space = self.env.action_space
        
        self.observation_space = self.env.observation_space if state_processor is None else gym.spaces.Box(low=-1, high=1, shape=(reduced_dim,))
        self.state_processor = state_processor
        self.safety = safety

        self._max_episode_steps = 500
       
        self.step_counter = 0
        self.done = False  
        self.safe_polys = []
        self.polys = []
        
        self.safety_constraints()
        self.unsafe_constraints()
        
        
    def safety_constraints(self):
        obs_space_lower = self.observation_space.low
        obs_space_upper = self.observation_space.high

        # Initialize lower and upper bounds as the observation space limits
        lower_bounds = np.copy(obs_space_lower)
        upper_bounds = np.copy(obs_space_upper)

        # Adjust the center for specific constraints
        center = (obs_space_lower + obs_space_upper) / 2

        # Horizontal position constraint (x) - relaxed
        lower_bounds[0] = center[0] - 0.75  # Increased from 0.75 to 1.0
        upper_bounds[0] = center[0] + 0.75

        # Vertical position constraint (y) - relaxed
        
        lower_bounds[1] = 0.0
        upper_bounds[1] = 1.75
        center[1] = (upper_bounds[1] + lower_bounds[1])/2

        # Horizontal velocity constraint (vx) - relaxed
        lower_bounds[2] = center[2] - 1.5  # Increased from 0.5 to 0.75
        upper_bounds[2] = center[2] + 1.5

        # Vertical velocity constraint (vy) - relaxed
        center[3] = -0.5  # Descent speed
        lower_bounds[3] = -1.5  # Increased from 0.5 to 0.75
        upper_bounds[3] = 0.5

        # Angle constraint (theta) - relaxed
        lower_bounds[4] = center[4] - 1.5  # Increased from 0.4 to 0.6
        upper_bounds[4] = center[4] + 1.5

        # Angular velocity constraint (omega) - relaxed
        lower_bounds[5] = center[5] - 1.0  # Increased from 0.3 to 0.5
        upper_bounds[5] = center[5] + 1.0

        polys = []
        center = (upper_bounds + lower_bounds)/2
        generators = [np.zeros(center.shape) for _ in range(center.shape[0])]

        # Create polyhedral constraints (polys) based on the bounds
        for i in range(self.observation_space.shape[0]):
            A1 = np.zeros(self.observation_space.shape[0])
            A2 = np.zeros(self.observation_space.shape[0])

            # Upper bound constraint: A[i] * x[i] <= u_i
            A1[i] = 1
            polys.append(np.append(A1, -upper_bounds[i]))

            # Lower bound constraint: A[i] * x[i] >= l_i, or -A[i] * x[i] <= -l_i
            A2[i] = -1
            polys.append(np.append(A2, lower_bounds[i]))
            generators[i][i] = (upper_bounds[i] - lower_bounds[i])/2

        # Set the safety constraints using the DeepPoly domain
        # input_deeppoly = domains.DeepPoly(lower_bounds, upper_bounds)
        input_deeppoly = domains.Zonotope(center, generators)
    
        self.original_safe_polys = [np.array(polys)]
        self.safe_polys = [np.array(polys)]
        self.safety = input_deeppoly
        self.original_safety = input_deeppoly
 
        
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
        # else:
            # state = self.reduce_state(state)
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
        # else:
            # state = self.reduce_state(state)
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
                return not np.all(A @ state.reshape(-1, 1) <= b.reshape(-1, 1))
    


