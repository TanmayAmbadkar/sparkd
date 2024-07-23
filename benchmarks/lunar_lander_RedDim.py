

#     #resets setting the legs to 0
#     def reset(self) -> np.ndarray:
#         self.state = self.init_space.sample()
#         self.state[6] = 0
#         self.state[7] = 0
#         self.step_counter = 0
#         reduced_state = self.reduce_state(self.state)
#         return reduced_state

#     def has_landed(self) -> bool:
#         _, y, vx, vy, _, _, leg_1, leg_2 = self.state
#         return y <= 0.1 and (leg_1 != 0 or leg_2 != 0) and abs(vx) < 0.05 and abs(vy) < 0.05

#     def has_crashed(self) -> bool:
#         _, y, vx, vy, _, _, _, _ = self.state
#         return y <= 0.1 and (abs(vx) > 0.5 or abs(vy) > 0.5)

#     def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
#         x, y, vx, vy, angle, angular_velocity, leg_1, leg_2 = self.state  # Unpack full state

#         main_throttle, lateral_throttle = action
#         thrust = main_throttle * 2 
#         lateral_thrust = lateral_throttle * 0.4  
#         vx += thrust * np.cos(angle) - lateral_thrust * np.sin(angle)
#         vy += thrust * np.sin(angle) + lateral_thrust * np.cos(angle) + self.gravity

#         if self.enable_wind:
#             vx += self.wind_power * np.random.randn()
#             vy += self.turbulence_power * np.random.randn()

#         x += vx
#         y += vy

#         # Update the full state
#         self.state = np.array([x, y, vx, vy, angle, angular_velocity, leg_1, leg_2])
        
#         target_x, target_y = 0, 0
#         distance = np.sqrt((x - target_x)**2 + (y - target_y)**2)
#         reward = -distance - (np.abs(vx) + np.abs(vy)) - np.abs(angle) * 10
#         reward -= 0.3 if main_throttle > 0 else 0
#         reward -= 0.03 if lateral_throttle != 0 else 0
#         reward += 100 if self.has_landed() else 0
#         reward -= 100 if self.has_crashed() else 0
#         reward -= 10 if abs(x) > 4 else 0
#         reward -= 10 if abs(vx) or abs(vy) > 3.5 else 0
#         reward += 10 if abs(vx) or abs(vy) < 1.5 else 0

#         if self.has_crashed():
#             print("The ship has crashed")
#             print("y", y, "vx", vx, "vy ", vy, "angle", angle)

#         if self.has_landed():
#             print("The ship has landed")

#         done = self.step_counter >= self._max_episode_steps or self.has_crashed() or self.has_landed()
#         self.step_counter += 1

#         # Return the reduced state
#         reduced_state = self.reduce_state(self.state)
#         return reduced_state, reward, done, {}  # Return the reduced state


#     def predict_done(self, state: np.ndarray) -> bool:
#         x, y, vx, vy, angle, angular_velocity = state 
        
#         landed = y <= 0.1 and abs(vx) < 0.1 and abs(vy) < 0.1
#         crashed = y <= 0.1 and (abs(vx) > 0.1 or abs(vy) > 0.1 or abs(angle) > 0.2)
#         out_of_bounds = np.abs(x) > 5 or np.abs(y) > 5
#         return landed or crashed or out_of_bounds

#     def unsafe(self, state: np.ndarray) -> bool:
#         x, y, vx, vy, angle, angular_velocity = state  
      
#         high_velocity_x = np.abs(vx) > 6.5
#         high_velocity_y = np.abs(vy) > 6.5
#         extreme_angle = np.abs(angle) > 0.25
#         low_altitude_unsafe = y < 0.1 and (np.abs(vx) > 0.2 or np.abs(angle) > 0.1)
#         print("Too fast sideways:", high_velocity_x)
#         print("Too fast down:", high_velocity_y)
#         print("Too much angle:", extreme_angle)
#         print("Too low and fast:", low_altitude_unsafe)
#         print(state)
#         return high_velocity_x or high_velocity_y or extreme_angle or low_altitude_unsafe

#     def seed(self, seed: int):
#         np.random.seed(seed)
#         self.action_space.seed(seed)
#         self.observation_space.seed(seed)
#         self.init_space.seed(seed)



import gymnasium as gym
import torch
import numpy as np

class LunarLanderEnv2(gym.Env):
    def __init__(self, state_processor=None, reduced_dim=None):
        self.env = gym.make("LunarLander-v2", continuous=True)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space if state_processor is None else gym.spaces.Box(low=-1, high=1, shape=(reduced_dim,))
        self.state_processor = state_processor

        self._max_episode_steps = 2000
       
        self.step_counter = 0
        self.done = False  

    def reduce_state(self, state: np.ndarray) -> np.ndarray:
        x, y, vx, vy, angle, angular_velocity = state[:6]
        newstate = np.array([x, y, vx, vy, angle, angular_velocity])
        return newstate

    def step(self, action):
        state, reward, done, truncation, info = self.env.step(action)
        # print("state: ", state)
        # print("reward: ", reward)
        # print("done: ", done)
        # print("truncation: ", truncation)
        # print("info: ", info)

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
        x, y, vx, vy, angle, angular_velocity = state
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

