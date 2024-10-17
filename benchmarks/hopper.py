import gymnasium as gym
import torch
import numpy as np
from abstract_interpretation import domains, verification

class HopperEnv(gym.Env):
    def __init__(self, state_processor=None, reduced_dim=None, safety=None):
        self.env = gym.make("Hopper-v5")


        self.action_space = self.env.action_space
        
        self.observation_space = self.env.observation_space if state_processor is None else gym.spaces.Box(low=-1, high=1, shape=(reduced_dim,))
        self.state_processor = state_processor
        self.safety = safety

        self._max_episode_steps = 1000  # You can adjust this as needed
        self.step_counter = 0
        self.done = False  
        self.safe_polys = []
        self.polys = []
        
        self.safety_constraints()
        self.unsafe_constraints()

    def safety_constraints(self):
        # Initialize the center and generators
        obs_shape = self.observation_space.shape[0]
        center = np.zeros(obs_shape)
        generators = []
        
        # Safe ranges based on your stats
        safe_min = [1.01, -0.2, -0.48, -0.75, -0.41, -1.51, -1.64, -5.69, -6.55, -6.49, -6.16]
        safe_max = [1.28, 0.18, 0.04, 0.04, 0.84, 1.82, 0.94, 5.26, 6.07, 5.38, 7.75]

        # Safe state boundaries based on Hopper termination conditions

        # safe_min = [
        #     0.8,  # z-coordinate of the torso (height), ensure it doesn't fall below 0.8m
        # -0.2,  # angle of the torso (radians), limit tilting of the torso
        # -0.5,  # angle of the thigh joint (radians)
        # -0.5,  # angle of the leg joint (radians)
        # -0.5,  # angle of the foot joint (radians)
        # -1.5,  # velocity of the x-coordinate of the torso (m/s)
        # -1.5,  # velocity of the z-coordinate of the torso (m/s), avoid sharp falls
        # -5.0,  # angular velocity of the torso (rad/s)
        # -5.0,  # angular velocity of the thigh hinge (rad/s)
        # -5.0,  # angular velocity of the leg hinge (rad/s)
        # -5.0   # angular velocity of the foot hinge (rad/s)
        # ]

        # safe_max = [
        #     1.25,  # z-coordinate of the torso (height), prevent hopper from jumping too high
        #     0.2,   # angle of the torso (radians), limit forward/backward tilt
        #     0.5,   # angle of the thigh joint (radians)
        #     0.5,   # angle of the leg joint (radians)
        #     0.5,   # angle of the foot joint (radians)
        #     2.0,   # velocity of the x-coordinate of the torso (m/s), prevent high-speed movement
        #     1.5,   # velocity of the z-coordinate of the torso (m/s), avoid excessive jumps
        #     5.0,   # angular velocity of the torso (rad/s), prevent rapid spinning
        #     5.0,   # angular velocity of the thigh hinge (rad/s)
        #     5.0,   # angular velocity of the leg hinge (rad/s)
        #     5.0    # angular velocity of the foot hinge (rad/s)
        # ]

        # Create the generators based on safe range limits
        for i in range(obs_shape):
            gen = np.zeros(obs_shape)
            center[i] = (safe_min[i] + safe_max[i]) / 2  # Midpoint for center
            gen[i] = (safe_max[i] - safe_min[i]) / 2  # Range width for generator
            generators.append(gen)
        
        # Convert to zonotope and store constraints
        input_zonotope = domains.Zonotope(center, generators)
        self.safety = input_zonotope
        self.original_safety = input_zonotope
        
        # Store polyhedra constraints
        polys = []
        for i, gen in enumerate(generators):
            cen = center[i]
            bound = gen[i]
            A1 = np.zeros(obs_shape)
            A2 = np.zeros(obs_shape)
            A1[i] = 1
            A2[i] = -1
            polys.append(np.append(A1, -(cen + bound)))
            polys.append(np.append(A2, (cen - bound)))
        
        self.original_safe_polys = [np.array(polys)]
        self.safe_polys = [np.array(polys)]

    def unsafe_constraints(self):
        unsafe_regions = []

        # Negate safe polys to define unsafe regions
        for polys in self.safe_polys:
            for poly in polys:
                A = poly[:-1]  # Coefficients
                b = -poly[-1]  # Constant term (negate)
                unsafe_regions.append(np.append(-A, b))  # Flip sign for unsafe regions
        
        # Extend unsafe regions if needed (e.g., beyond boundaries)
        for i in range(self.observation_space.shape[0]):
            A1 = np.zeros(self.observation_space.shape[0])
            A2 = np.zeros(self.observation_space.shape[0])
            A1[i] = 1
            A2[i] = -1
            unsafe_regions.append(np.append(A1, -10))  # Extend to large negative bounds
            unsafe_regions.append(np.append(A2, -10))  # Extend to large positive bounds
        
        self.polys = [np.array(unsafe_regions)]

    def step(self, action):
        state, reward, done, truncation, info = self.env.step(action)
        self.done = done or self.step_counter >= self._max_episode_steps

        original_state = np.copy(state)
        if self.state_processor is not None:
            state = self.state_processor(state.reshape(1, -1))
            state = state.reshape(-1,)
        self.step_counter += 1
        return state, reward, self.done, truncation, {"state_original": original_state}

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)

        self.step_counter = 0
        self.done = False 
        original_state = np.copy(state)
        if self.state_processor is not None:
            state = self.state_processor(state.reshape(1, -1))
            state = state.reshape(-1,)
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

    def unsafe(self, state: np.ndarray, simulated: bool = False) -> bool:
        if simulated:
            for polys in self.safe_polys:
                A = polys[:, :-1]
                b = -polys[:, -1]
                return not np.all(A @ state.reshape(-1, 1) <= b.reshape(-1, 1))
        else:
            for polys in self.original_safe_polys:
                A = polys[:, :-1]
                b = -polys[:, -1]
                # if not np.all(A @ state.reshape(-1, 1) <= b.reshape(-1, 1)):
                #     temp = A @ state.reshape(-1, 1) <= b.reshape(-1, 1)
                #     temp = np.bitwise_not(temp)
                #     print(A[temp.reshape(-1, )])
                #     print(b[temp.reshape(-1, )])
                #     print(state)
                #     print(temp)
                return not np.all(A @ state.reshape(-1, 1) <= b.reshape(-1, 1))
                # return False
