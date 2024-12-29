import gymnasium as gym
import numpy as np
from pytorch_soft_actor_critic.replay_memory import ReplayMemory
class StableBaselinesEnv(gym.Env):
    
    def __init__(self, real_env, replay_buffer: ReplayMemory = None):
        
        self.real_env = real_env
        self.action_space = self.real_env.action_space
        self.observation_space = self.real_env.observation_space
        self.sim_env = None
        self.real_flag = True
        
        self.curr_state = None
        self.curr_steps = 0
        self.replay_buffer = replay_buffer
    
    def step(self, action):
        
        if self.real_flag:
            state, reward, done, trunc, info = self.real_env.step(action)
            if self.real_env.unsafe(state):
                done = True
                reward += - 10
            
            self.replay_buffer.push(self.curr_state, action, reward, state, done, int(self.real_env.unsafe(state)))
            self.curr_state = state
            return state, reward, done, trunc, info
        else:
            next_state, reward = self.sim_env(self.curr_state, action, use_neural_model = False)
            done = not np.all(np.abs(next_state) < 1e5) and \
                not np.any(np.isnan(next_state))
            # done = done or env.pred`ict_done(next_state)
            done = done or self.curr_steps == self.real_env._max_episode_steps or \
                not np.all(np.abs(next_state) < 1e5)
                
            self.curr_state = next_state
            return next_state, reward, done, False, {}
        
    def reset(self, seed = None):
        
        state, info = self.real_env.reset(seed = seed)
        self.curr_state = state
        return state, info
            