import gymnasium as gym
import torch
import numpy as np
import sys

class CheetahEnv(gym.Env):
    def __init__(self, state_processor=None, reduced_dim=None, safety=None):
        self.env = gym.make("HalfCheetah-v5")
        self.action_space = self.env.action_space
        
        self.observation_space = self.env.observation_space if state_processor is None else gym.spaces.Box(low=-1, high=1, shape=(reduced_dim,))
        self.state_processor = state_processor
        self.safety = safety

        self._max_episode_steps = 1000
       
        self.step_counter = 0
        self.done = False  
        self.safe_polys = []
        self.polys = []
        
        
        # print(self.unsafe(np.array([ 0.41278508,  0.11044428,  0.03596416, -0.0501044,  -0.520235,   -0.7669368,
        #         0.55146146, -1.,          0.,         -0.3183163,  -1.0000002,   0.109326,
        #         0.9999997,   0.,          0.46180838,  0.4670529,   0.48339868,  0.51286566,
        #         0.55954015,  0.63115406,  0.7429231,   0.92812556,  1.,          1.,        ])))
        # sys.exit()
            

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
        
        return state, reward, self.done, {"state_original": original_state}

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

    def unsafe(self, state: np.ndarray, simulated:bool = False) -> bool:
        
        return not -2.8795 <= state[9] <= 2.8795
    

