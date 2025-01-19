import gymnasium as gym

class LearnedEnv(gym.Env):
    
    def __init__(self, original_env, env_model, obs_space_domain):
        
        self.original_env = original_env
        self.env_model = env_model
        self.obs_space_domain = obs_space_domain
        self.observation_space = gym.spaces.Box(low=obs_space_domain.lower.numpy(), high=obs_space_domain.upper.numpy(), shape=(obs_space_domain.upper.numpy().shape[0],))
        self.action_space = original_env.action_space
        self.curr_state = None
        self.curr_steps = 0
        self._max_episode_steps = 500
        
    def step(self, action):
        
        next_state, reward = self.env_model(self.curr_state, action,
                                           use_neural_model=False)
        
        done, trunc = False, False
        if self.original_env.unsafe(next_state, simulated = True):
            done = True
            reward = -100
            
        if self.curr_steps == self._max_episode_steps:
            trunc = True
        
        self.curr_state = next_state
        return next_state, reward, done, trunc, {"is_success": self.original_env.unsafe(next_state, simulated = True)}