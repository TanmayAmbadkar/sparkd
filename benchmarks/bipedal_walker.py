import gymnasium as gym
import torch
import numpy as np
from abstract_interpretation import domains, verification
import sys

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

        # Calculate the center of the zonotope
        center = (obs_space_lower + obs_space_upper) / 2

        # Create generators to reflect the constraints on vx and vy
        generators = []
        
        # Angular Speed
        hull_ang_gen = np.zeros(self.observation_space.shape[0])
        hull_ang_gen[0] = 0.7

        vang_gen = np.zeros(self.observation_space.shape[0])
        vang_gen[1] = 0.9
        center[1] = 0.25

        vhor_gen = np.zeros(self.observation_space.shape[0])
        vhor_gen[2] = 0.8

        vver_gen = np.zeros(self.observation_space.shape[0])
        vver_gen[3] = 0.8
        
        hipjoinang_gen = np.zeros(self.observation_space.shape[0])
        hipjoinang_gen[4] = 1.1
        center[4] = 0.15

        
        # Add these generators to the list
        generators.append(hull_ang_gen)
        generators.append(vang_gen)
        generators.append(vhor_gen)
        generators.append(vver_gen)
        generators.append(hipjoinang_gen)

        # Additional generators for other dimensions can be added if necessary
        # Example: small perturbations in other dimensions

        polys = []
        for i, gen in enumerate(generators):
            
            cen = center[i]
            bound = gen[i]
            A1 = np.zeros(self.observation_space.shape[0])
            A2 = np.zeros(self.observation_space.shape[0])
            A1[i] = 1
            A2[i] = -1
            polys.append(np.append(A1, -(cen + bound)))
            polys.append(np.append(A2, (cen - bound)))
             
        
        for i in range(self.observation_space.shape[0]):
            if i not in [0, 1, 2, 3, 4]:
                # if i in range(14, 24, 1):
                #     gen = np.zeros(self.observation_space.shape[0])
                #     gen[i] = 0.9
                #     center[i] = 0
                #     generators.append(gen)
                # else:
                gen = np.zeros(self.observation_space.shape[0])
                gen[i] = (obs_space_upper[i] - obs_space_lower[i]) / 2
                generators.append(gen)
                
                A1 = np.zeros(self.observation_space.shape[0])
                A2 = np.zeros(self.observation_space.shape[0])
                A1[i] = 1
                A2[i] = -1
                polys.append(np.append(A1, -obs_space_upper[i]))
                polys.append(np.append(A2, obs_space_lower[i]))
                
        # hyperplanes = input_zonotope.to_hyperplanes()
        # for i, poly in enumerate(polys):
        #     A = poly[:-1]
        #     b = -poly[-1]
        #     print(f"x_{i//2} {'<=' if A[i//2] > 0 else '>='} {b/sum(A)}")
            # polys.append(np.append(A, -b))        

        # # Create the zonotope
        input_zonotope = domains.Zonotope(center, generators)
        
        self.original_safe_polys = [np.array(polys)]
        self.safe_polys = [np.array(polys)]
        self.safety = input_zonotope
        self.original_safety = input_zonotope
        
        
    def unsafe_constraints(self):
        
        obs_space_lower = self.observation_space.low
        obs_space_upper = self.observation_space.high
        unsafe_regions = []
        for polys in self.safe_polys:
            for i, poly in enumerate(polys):
                if i//2 in [0, 1, 2, 3, 4]:
                    A = poly[:-1]
                    b = -poly[-1]
                    unsafe_regions.append(np.append(-A, b))
            # print(f"x_{i//2} {'<=' if A[i//2] > 0 else '>='} {1/sum(A/-b)}")
                # else:
                #     A1 = np.zeros(self.observation_space.shape[0])
                #     A2 = np.zeros(self.observation_space.shape[0])
                #     A1[i//2] = 1
                #     A2[i//2] = -1
                #     unsafe_regions.append(np.append(A1, -obs_space_upper[i//2]))
                #     unsafe_regions.append(np.append(A2, obs_space_lower[i//2]))
        
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
                # print(A @ state.reshape(-1, 1) <= b.reshape(-1, 1))
                return not np.all(A @ state.reshape(-1, 1) <= b.reshape(-1, 1))
    


