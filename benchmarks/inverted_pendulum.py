import gymnasium as gym
import torch
import numpy as np
from constraints import safety, verification

class InvertedPendulumEnv(gym.Env):
    def __init__(self, state_processor=None, reduced_dim=None, safety=None):
        self.env = gym.make("InvertedPendulum-v5")


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

        # Initialize lower and upper bounds as the observation space limits
        lower_bounds = np.copy(obs_space_lower)
        upper_bounds = np.copy(obs_space_upper)

        # Adjust the center for specific constraints
        center = (obs_space_lower + obs_space_upper) / 2

        # Horizontal position constraint (x) - relaxed
        lower_bounds[0] = center[0] - 0.5  # Allow larger movement along x-axis
        upper_bounds[0] = center[0] + 0.5

        # Sine of the angle between the cart and the first pole (relaxed)
        lower_bounds[1] = center[1] - 0.75  # Allow larger variations in the angle
        upper_bounds[1] = center[1] + 0.75

        # Sine of the angle between the two poles
        lower_bounds[2] = center[2] - 0.75  # Allow larger variations between poles
        upper_bounds[2] = center[2] + 0.75

        # Cosine of the angle between the cart and the first pole
        center[3] = 0.825  # Cosine close to 1 when the pole is upright
        lower_bounds[3] = center[3] - 0.175
        upper_bounds[3] = center[3] + 0.175

        # Cosine of the angle between the two poles
        center[4] = 0.8  # Cosine close to 1 when poles are aligned
        lower_bounds[4] = center[4] - 0.2
        upper_bounds[4] = center[4] + 0.2

        # Velocity of the cart
        lower_bounds[5] = center[5] - 5.0  # Allow the cart to move faster
        upper_bounds[5] = center[5] + 5.0

        # Angular velocity of the angle between the cart and the first pole
        lower_bounds[6] = center[6] - 8.0  # Allow higher angular velocity for the first pole
        upper_bounds[6] = center[6] + 8.0

        # Angular velocity of the angle between the two poles
        lower_bounds[7] = center[7] - 7.5  # Allow higher angular velocity for the second pole
        upper_bounds[7] = center[7] + 7.5

        # Force constraint
        lower_bounds[8] = center[8] - 9.0
        upper_bounds[8] = center[8] + 9.0

        polys = []
        center = (upper_bounds + lower_bounds) / 2
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
            generators[i][i] = (upper_bounds[i] - lower_bounds[i]) / 2

        # Set the safety constraints using the Box domain
        # input_Box = domains.Box(lower_bounds, upper_bounds)
        input_Box = safety.Zonotope(center, generators)

        self.original_safe_polys = [np.array(polys)]
        self.safe_polys = [np.array(polys)]
        self.safety = input_Box
        self.original_safety = input_Box



    def unsafe_constraints(self):
        
        obs_space_lower = self.observation_space.low
        obs_space_upper = self.observation_space.high
        unsafe_regions = []

        # Define unsafe regions based on the safe polyhedra
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