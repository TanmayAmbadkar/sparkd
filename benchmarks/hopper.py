import gymnasium as gym
import torch
import numpy as np
from abstract_interpretation import domains, verification
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

        # Set the new bounds for each relevant component
        # Set the bounds for each component based on provided MIN and MAX
        lower_bounds[0] = 0.7   # MIN
        upper_bounds[0] = 9   # MAX

        lower_bounds[1] = -0.2   # MIN
        upper_bounds[1] = 0.2   # MAX

        lower_bounds[2] = -0.34  # MIN
        upper_bounds[2] = 0.04   # MAX

        lower_bounds[3] = -1.07  # MIN
        upper_bounds[3] = 0.03   # MAX

        lower_bounds[4] = -0.43  # MIN
        upper_bounds[4] = 0.83   # MAX

        lower_bounds[5] = -1.33  # MIN
        upper_bounds[5] = 1.47   # MAX

        lower_bounds[6] = -1.07  # MIN
        upper_bounds[6] = 4.51   # MAX

        lower_bounds[7] = -5.22  # MIN
        upper_bounds[7] = 4.94   # MAX

        lower_bounds[8] = -6.01  # MIN
        upper_bounds[8] = 5.84   # MAX

        lower_bounds[9] = -6.41  # MIN
        upper_bounds[9] = 5.25   # MAX

        lower_bounds[10] = -5.51 # MIN
        upper_bounds[10] = 6.82  # MAX
        
        center = (upper_bounds + lower_bounds)/2

        # Construct the polyhedra constraints (polys)
        polys = []
        center = (upper_bounds + lower_bounds)/2
        generators = [np.zeros(center.shape) for _ in range(center.shape[0])]
        for i in range(self.observation_space.shape[0]):
            # Upper bound constraint: A[i] * x[i] <= u_i
            A_upper = np.zeros(self.observation_space.shape[0])
            A_upper[i] = 1
            polys.append(np.append(A_upper, -upper_bounds[i]))

            # Lower bound constraint: A[i] * x[i] >= l_i, or -A[i] * x[i] <= -l_i
            A_lower = np.zeros(self.observation_space.shape[0])
            A_lower[i] = -1
            polys.append(np.append(A_lower, lower_bounds[i]))
            generators[i][i] = (upper_bounds[i] - lower_bounds[i])/2
            

        # Create the DeepPolyDomain using the derived lower and upper bounds
        # input_deeppoly_domain = domains.DeepPoly(lower_bounds, upper_bounds)
        input_deeppoly_domain = domains.Zonotope(center, generators)

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
            # state = torch.Tensor(state, )
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
                # print(A @ state.reshape(-1, 1) <= b.reshape(-1, 1))
                return not np.all(A @ state.reshape(-1, 1) <= b.reshape(-1, 1))
    


