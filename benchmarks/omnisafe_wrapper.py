import gymnasium as gym

class OmnisafeWrapper(gym.Env):
    
    def __init__(self, env):
        
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, seed = None):
        
        return self.env.reset(seed)
    
    def step(self, action):
        
        state, reward, done, trunc, info = self.env.step(action)
        if self.env.unsafe(state):
            cost = 1
        else:
            cost = 0
            
        return state, reward, cost, done, trunc, info