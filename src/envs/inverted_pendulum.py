import gymnasium as gym
import torch
import numpy as np
# from constraints import safety

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
        
        # self.safety_constraints()
        # self.unsafe_constraints()

    # def safety_constraints(self):
    #     pass



    # def unsafe_constraints(self):
    #     pass

    def step(self, action):
        state, reward, done, truncation, info = self.env.step(action)
        self.done = done or self.step_counter >= self._max_episode_steps

        original_state = np.copy(state)
        if self.state_processor is not None:
            state = self.state_processor(state.reshape(1, -1))
            state = state.reshape(-1,)
        self.step_counter += 1
        cost = 1.0 if hasattr(self, 'unsafe') and self.unsafe(state) else 0.0
        return  state,  reward, cost,  self.done,  truncation,  {}

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)

        self.step_counter = 0
        self.done = False 
        original_state = np.copy(state)
        if self.state_processor is not None:
            state = self.state_processor(state.reshape(1, -1))
            state = state.reshape(-1,)
        return state, {}

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