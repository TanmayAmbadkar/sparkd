import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any
import torch
from abstract_interpretation import domains, verification



class AccEnv(gym.Env):

    def __init__(self):
        super(AccEnv, self).__init__()

        self.action_space = gym.spaces.Box(low=-2, high=2, shape=(1,))
        self.observation_space = gym.spaces.Box(low=np.array([-10, -10]),
                                                high=np.array([10, 10]))

        self.init_space = gym.spaces.Box(low=np.array([-1.1, -0.1]),
                                         high=np.array([-0.9, 0.1]))
        self.state = np.zeros(2)

        self.rng = np.random.default_rng()

        self.concrete_safety = [
            lambda x: x[0]
        ]

        self._max_episode_steps = 300

        self.polys = [
            # P (s 1)^T <= 0 iff s[0] >= 0
            # P = [[-1, 0, 0]]
            np.array([[-1.0, 0.0, 0.0]])
        ]

        self.safe_polys = [
            np.array([[1.0, 0.0, 0.01]])
        ]

        self.original_observation_space = self.observation_space
        self.continuous = True
        
        self.observation_space = self.observation_space
        self.state_processor = None
        self.safety = None

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

        input_deeppoly = domains.DeepPoly(lower_bounds, upper_bounds)
    
        self.original_safe_polys = [np.array(input_deeppoly.to_hyperplanes())]
        self.safe_polys = [np.array(input_deeppoly.to_hyperplanes())]
        self.safety = input_deeppoly
        self.original_safety = input_deeppoly
 
        
    def unsafe_constraints(self):
        
        
        unsafe_deeppolys = domains.recover_safe_region(domains.DeepPoly(self.observation_space.low, self.observation_space.high), [self.original_safety])        
        self.polys = []
        self.unsafe_domains = unsafe_deeppolys
        
        
        for poly in unsafe_deeppolys:
            self.polys.append(np.array(poly.to_hyperplanes()))


    def reset(self) -> np.ndarray:
        self.state = self.init_space.sample()
        self.steps = 0
        return self.state

    def step(self, action: np.ndarray) -> \
            Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        dt = 0.02
        x = self.state[0] + dt * self.state[1]
        v = self.state[1] + dt * \
            (action[0] + self.rng.normal(loc=0, scale=0.5))
        self.state = np.clip(
                np.asarray([x, v]),
                self.observation_space.low,
                self.observation_space.high)
        reward = 2.0 + x if x < 0 else -10
        done = bool(x >= 0) or self.steps > self._max_episode_steps
        self.steps += 1
        return self.state, reward, done, {}

    def predict_done(self, state: np.ndarray) -> bool:
        return state[0] >= 0

    def seed(self, seed: int):
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self.init_space.seed(seed)
        self.rng = np.random.default_rng(np.random.PCG64(seed))

    def unsafe(self, state: np.ndarray) -> bool:
        return state[0] >= 0
