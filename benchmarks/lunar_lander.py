import gymnasium as gym
import torch
import numpy as np
from abstract_interpretation import domains, verification
import sys

class LunarLanderEnv(gym.Env):
    def __init__(self, state_processor=None, reduced_dim=None, safety=None):
        self.env = gym.make("LunarLander-v2", continuous=True)
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
        center = (obs_space_lower + obs_space_upper) / 2

        generators = []

        # Horizontal position constraint (x)
        x_gen = np.zeros(self.observation_space.shape[0])
        x_gen[0] = 0.75
        generators.append(x_gen)

        # Vertical position constraint (y)
        y_gen = np.zeros(self.observation_space.shape[0])
        center[1] = 0.75  # Safe height above ground
        y_gen[1] = 0.75
        generators.append(y_gen)

        # Horizontal velocity constraint (vx)
        vx_gen = np.zeros(self.observation_space.shape[0])
        vx_gen[2] = 0.5  # Limit horizontal drift speed
        generators.append(vx_gen)

        # Vertical velocity constraint (vy)
        vy_gen = np.zeros(self.observation_space.shape[0])
        center[3] = -0.2  # Descent speed
        vy_gen[3] = 0.5   # Prevent rapid descent
        generators.append(vy_gen)

        # Angle constraint (theta)
        angle_gen = np.zeros(self.observation_space.shape[0])
        angle_gen[4] = 0.4  # Slight tilting allowed
        generators.append(angle_gen)

        # Angular velocity constraint (omega)
        angular_velocity_gen = np.zeros(self.observation_space.shape[0])
        angular_velocity_gen[5] = 0.3  # Limit rotational speed
        generators.append(angular_velocity_gen)

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
            if i not in [0, 1, 2, 3, 4, 5]:
                gen = np.zeros(self.observation_space.shape[0])
                gen[i] = (obs_space_upper[i] - obs_space_lower[i]) / 2
                generators.append(gen)
                
                A1 = np.zeros(self.observation_space.shape[0])
                A2 = np.zeros(self.observation_space.shape[0])
                A1[i] = 1
                A2[i] = -1
                polys.append(np.append(A1, -obs_space_upper[i]))
                polys.append(np.append(A2, obs_space_lower[i]))
                
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
                if i//2 in [0, 1, 2, 3, 4, 5]:
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
                # print(A @ state.reshape(-1, 1) <= b.reshape(-1, 1))
                return not np.all(A @ state.reshape(-1, 1) <= b.reshape(-1, 1))
    


