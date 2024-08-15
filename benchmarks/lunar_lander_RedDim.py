import gymnasium as gym
import torch
import numpy as np

class LunarLanderEnv2(gym.Env):
    def __init__(self, state_processor=None, reduced_dim=None, safety=None):
        self.env = gym.make("LunarLander-v2", continuous=True)
        self.action_space = self.env.action_space
        
        # Forced  size for lowering dim 
        self.observation_space = gym.spaces.Box(low = np.array([-1.5, -1.5, -5., -5., -3.1415927, -5., ]), high = np.array([1.5, 1.5, 5., 5., 3.1415927, 5.,]), shape = (6,), ) if state_processor is None else gym.spaces.Box(low=-1, high=1, shape=(reduced_dim,))

        # self.observation_space = self.env.observation_space if state_processor is None else gym.spaces.Box(low=-1, high=1, shape=(reduced_dim,))
        self.state_processor = state_processor
        self.safety = safety

        self._max_episode_steps = 500
       
        self.step_counter = 0
        self.done = False  
         

    def reduce_state(self, state: np.ndarray) -> np.ndarray:
        x, y, vx, vy, angle, angular_velocity = state[:6]
        newstate = np.array([x, y, vx, vy, angle, angular_velocity])
        return newstate

    def step(self, action):
        
        state, reward, done, truncation, info = self.env.step(action)
        self.done = done  # Store the done flag

        if self.state_processor is not None:
            state = torch.Tensor(state)
            with torch.no_grad():
                state = self.state_processor(state.reshape(1, -1))
            state = state.numpy()
        else:
            state = self.reduce_state(state)
        
        return state, reward, done, truncation

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)

        self.step_counter = 0
        self.done = False 

        if self.state_processor is not None:
            state = torch.Tensor(state)
            with torch.no_grad():
                state = self.state_processor(state.reshape(1, -1))
            state = state.numpy()
        else:
            state = self.reduce_state(state)
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
    


