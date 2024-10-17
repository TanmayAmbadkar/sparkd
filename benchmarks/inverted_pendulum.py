import gymnasium as gym
import torch
import numpy as np
from abstract_interpretation import domains, verification

class InvertedDoublePendulumEnv(gym.Env):
    def __init__(self, state_processor=None, reduced_dim=None, safety=None):
        self.env = gym.make("InvertedDoublePendulum-v5")


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
        
        obs_space_lower = self.observation_space.low
        obs_space_upper = self.observation_space.high

        # Define the center of the zonotope
        center = np.zeros(shape=self.observation_space.shape)

        # Create generators for constraints (position, angles, velocities)
        generators = []
        
        # Constraint 1: Position of the cart (allow more freedom of movement along x-axis)
        position_gen = np.zeros(self.observation_space.shape[0])
        position_gen[0] = 0.5 # Allow larger movement along x-axis (less strict)
        center[0] = 0.0  # Keep the cart centered
        generators.append(position_gen)

        # Constraint 2: Sine of the angle between the cart and the first pole (allow more freedom)
        angle1_sin_gen = np.zeros(self.observation_space.shape[0])
        angle1_sin_gen[1] = 0.75  # Allow larger variations in the angle
        center[1] = 0.0  # Keep angle close to vertical
        generators.append(angle1_sin_gen)

        # Constraint 3: Sine of the angle between the two poles
        angle2_sin_gen = np.zeros(self.observation_space.shape[0])
        angle2_sin_gen[2] = 0.75  # Allow larger variations between poles
        center[2] = 0.0  # Keep angle between poles stable
        generators.append(angle2_sin_gen)

        # Constraint 4: Cosine of the angle between the cart and the first pole
        angle1_cos_gen = np.zeros(self.observation_space.shape[0])
        angle1_cos_gen[3] = 0.175  # Allow more freedom in cosine value
        center[3] = 0.825  # Cosine close to 1 when the pole is upright
        generators.append(angle1_cos_gen)

        # Constraint 5: Cosine of the angle between the two poles
        angle2_cos_gen = np.zeros(self.observation_space.shape[0])
        angle2_cos_gen[4] = 0.2  # Allow more freedom in cosine value between poles
        center[4] = 0.8  # Cosine close to 1 when poles are aligned
        generators.append(angle2_cos_gen)

        # Constraint 6: Velocity of the cart (allow faster movement)
        velocity_gen = np.zeros(self.observation_space.shape[0])
        velocity_gen[5] = 5.0  # Allow the cart to move faster
        center[5] = 0.0  # Keep the cart's velocity within reasonable bounds
        generators.append(velocity_gen)

        # Constraint 7: Angular velocity of the angle between the cart and the first pole
        angular_velocity1_gen = np.zeros(self.observation_space.shape[0])
        angular_velocity1_gen[6] = 8.0  # Allow higher angular velocity for the first pole
        center[6] = 0.0  # Keep angular velocity around 0
        generators.append(angular_velocity1_gen)

        # Constraint 8: Angular velocity of the angle between the two poles
        angular_velocity2_gen = np.zeros(self.observation_space.shape[0])
        angular_velocity2_gen[7] = 7.5 # Allow higher angular velocity for the second pole
        center[7] = 0.0  # Keep angular velocity around 0
        generators.append(angular_velocity2_gen)
        
        force = np.zeros(self.observation_space.shape[0])
        force[8] = 9.0  # Allow higher angular velocity for the second pole
        center[8] = 0.0  # Keep angular velocity around 0
        generators.append(force)
        
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


        # Create the zonotope for the safe region
        input_zonotope = domains.Zonotope(center, generators)
        self.original_safe_polys = [np.array(polys)]
        self.safe_polys = [np.array(polys)]
        self.safety = input_zonotope
        self.original_safety = input_zonotope
        
        


    def unsafe_constraints(self):
        
        obs_space_lower = self.observation_space.low
        obs_space_upper = self.observation_space.high
        unsafe_regions = []

        # Define unsafe regions based on the safe polyhedra
        for polys in self.safe_polys:
            for i, poly in enumerate(polys):
                if i // 2 in [0, 1, 2, 3, 4, 5, 6, 7]:  # Pendulum angles, velocities, etc.
                    A = poly[:-1]
                    b = -poly[-1]
                    unsafe_regions.append(np.append(-A, b))

        for i in range(self.observation_space.shape[0]):
            A1 = np.zeros(self.observation_space.shape[0])
            A2 = np.zeros(self.observation_space.shape[0])
            A1[i] = 1
            A2[i] = -1
            unsafe_regions.append(np.append(A1, -100))
            unsafe_regions.append(np.append(A2, -100))

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
