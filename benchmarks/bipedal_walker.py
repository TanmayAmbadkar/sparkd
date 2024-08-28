import gymnasium as gym
import torch
import numpy as np
from abstract_interpretation import domains, verification

class BipedalWalkerEnv(gym.Env):
    def __init__(self, state_processor=None, reduced_dim=None, safety=None):
        self.env = gym.make("BipedalWalker-v3", continuous=True)
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
        hull_ang_gen[0] = 3

        vang_gen = np.zeros(self.observation_space.shape[0])
        vang_gen[1] = 2

        vhor_gen = np.zeros(self.observation_space.shape[0])
        vhor_gen[2] = 3

        vver_gen = np.zeros(self.observation_space.shape[0])
        vver_gen[3] = 3

        
        # Add these generators to the list
        generators.append(hull_ang_gen)
        generators.append(vang_gen)
        generators.append(vhor_gen)
        generators.append(vver_gen)

        # Additional generators for other dimensions can be added if necessary
        # Example: small perturbations in other dimensions
        for i in range(self.observation_space.shape[0]):
            if i not in [0, 1, 2, 3]:
                gen = np.zeros(self.observation_space.shape[0])
                gen[i] = (obs_space_upper[i] - obs_space_lower[i]) / 2
                generators.append(gen)

        # Create the zonotope
        input_zonotope = domains.Zonotope(center, generators)

        polys = []
        print("Final Zonotope details")
        hyperplanes = input_zonotope.to_hyperplanes()
        for A, b in hyperplanes:
            print(f"Hyperplane: {A} * x <= {b}")
            polys.append(np.append(A, b))
            
        self.polys = polys
        self.safety = input_zonotope
        
    def reduce_state(self, state: np.ndarray) -> np.ndarray:
        x, y, vx, vy, angle, angular_velocity, a, b = state[:8]
        newstate = np.array([x, y, vx, vy, angle, angular_velocity, a, b])
        
        return newstate

    def step(self, action):
        
        state, reward, done, truncation, info = self.env.step(action)
        self.done = done  # Store the done flag

        if self.state_processor is not None:
            # state = self.reduce_state(state)
            # state = torch.Tensor(state, dtype = torch.float64)
            with torch.no_grad():
                state = self.state_processor(state.reshape(1, -1))
            # state = state.numpy()
            state = state.reshape(-1,)
        # else:
            # state = self.reduce_state(state)
        
        return state, reward, done, truncation

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)

        self.step_counter = 0
        self.done = False 

        if self.state_processor is not None:
            # state = self.reduce_state(state)
            # state = torch.Tensor(state)
            with torch.no_grad():
                state = self.state_processor(state.reshape(1, -1))
            # state = state.numpy()
            state = state.reshape(-1,)
        # else:
            # state = self.reduce_state(state)
        return state

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

    def unsafe(self, state: np.ndarray) -> bool:
        
        return not self.safety.in_zonotope(state)
        
        # return high_velocity_x or high_velocity_y
    


