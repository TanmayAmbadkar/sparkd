
import gymnasium as gym
import torch
import numpy as np

class LunarLanderEnv(gym.Env):
    def __init__(self, state_processor=None, reduced_dim=None):
        self.env = gym.make("LunarLander-v2", continuous=True)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space if state_processor is None else gym.spaces.Box(low=-1, high=1, shape=(reduced_dim,))
        self.state_processor = state_processor

        self._max_episode_steps = 2000

        
        self.step_counter = 0
        self.done = False  


    def reduce_state(self, state: np.ndarray) -> np.ndarray:
        # print("state: ", state)
        return state

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
        
        # Return values without the info dictionary
        return state, reward, done, truncation

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)

        self.step_counter = 0
        self.done = False  # Reset the done flag



        if self.state_processor is not None:
            state = torch.Tensor(state)
            with torch.no_grad():
                state = self.state_processor(state.reshape(1, -1))
            state = state.numpy()
        else:
            state = self.reduce_state(state)
        
        # Return state only, ignore info
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
        x, y, vx, vy, angle, angular_velocity, leg_1, leg_2 = state
        high_velocity_x = np.abs(vx) > 2
        high_velocity_y = np.abs(vy) > 2
        # extreme_angle = np.abs(angle) > 0.25
        # low_altitude_unsafe = y < 0.1 and (np.abs(vx) > 0.2 or np.abs(angle) > 0.1)
        # print("Too fast sideways: ", high_velocity_x)
        # print("Too fast down: ", high_velocity_y)
        # # print("Too much angle: ", extreme_angle)
        # # print("Too low and fast: ", low_altitude_unsafe)
        # print(state)
        return high_velocity_x or high_velocity_y #or extreme_angle or low_altitude_unsafe

