import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any

class LunarLanderEnv(gym.Env):
    def __init__(self, gravity=-10.0, enable_wind=False, wind_power=0.5, turbulence_power=0.5):
        super().__init__()

        # Constants
        self.gravity = gravity
        self.enable_wind = enable_wind
        self.wind_power = wind_power
        self.turbulence_power = turbulence_power

        # Action space: Continuous, Main engine and lateral boosters
        self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

        # Observation space: x, y, vx, vy, angle, angular velocity
        # Might need to be changed when implementing dimensionality reduction techniques
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

        # Initial conditions space: x, y, vx, vy, angle, angular velocity, leg 1, leg 2
        self.init_space = gym.spaces.Box(low=np.array([-0.5, -0.5, -0.1, -0.1, -0.1, -0.1,0,0]), 
                                         high=np.array([0.5, 0.5, 0.1, 0.1, 0.1, 0.1,1,1]), dtype=np.float32)

        # Simulation parameters
        self._max_episode_steps = 400
        self.step_counter = 0

# adapted from code below:

# def check_safety(state):
#     # 0 x, 1 y, 2 x_v, 3 y_v, 4 A, 5 A_v, 6 leg_1, 7 leg_2
#     y_pos = state[0][1]
#     y_vel = state[0][3]
#     angle = state[0][4]
#     angular_vel = state[0][5]
#     # random not specific really
#     max_y_vel = 3.5
#     max_angle = .20
#     max_angular_vel = 0.5
#     safe_y_vel = abs(y_vel) <= max_y_vel * (1 - np.exp(-abs(y_pos))) #safe y velocity
#     safe_angle = abs(angle) <= max_angle #angle 
#     safe_angular_vel = abs(angular_vel) <= max_angular_vel #angle velocity
#     return safe_y_vel and safe_angle and safe_angular_vel

#Ax=b
        self.polys = [np.array([
              [0, 1, 0, 0, 0, 0, 0, 0, -3.5],      # -y_vel <= -3.5 (y_vel >= -3.5)
              [0, -1, 0, 0, 0, 0, 0, 0, -3.5],     # y_vel <= 3.5
              [0, 0, 0, 0, 1, 0, 0, 0, -0.20],     # -angle <= -0.20 (angle >= -0.20)
              [0, 0, 0, 0, -1, 0, 0, 0, -0.20],    # angle <= 0.20
              [0, 0, 0, 0, 0, 1, 0, 0, -0.5],      # -angular_vel <= -0.5 (angular_vel >= -0.5)
              [0, 0, 0, 0, 0, -1, 0, 0, -0.5]])    # angular_vel <= 0.5
]
        self.safe_polys = [np.array([
              [0, 1, 0, 0, 0, 0, 0, 0, -4.0],      # -y_vel <= -4.0 (y_vel >= -4.0)
              [0, -1, 0, 0, 0, 0, 0, 0, -4.0],     # y_vel <= 4.0
              [0, 0, 0, 0, 1, 0, 0, 0, -0.25],     # -angle <= -0.25 (angle >= -0.25)
              [0, 0, 0, 0, -1, 0, 0, 0, -0.25],    # angle <= 0.25
              [0, 0, 0, 0, 0, 1, 0, 0, -0.6],      # -angular_vel <= -0.6 (angular_vel >= -0.6)
              [0, 0, 0, 0, 0, -1, 0, 0, -0.6]])    # angular_vel <= 0.6
]




    def reset(self) -> np.ndarray:
        self.state = self.init_space.sample()
        self.step_counter = 0

        return self.state
    
    def has_landed(self):
        # Simple check for landing, needs to be defined based on your simulation parameters
        x, y, vx, vy, angle, angular_velocity, leg_1, leg_2 = self.state
        return y <= 0 and leg_1 and leg_2

    def has_crashed(self):
        # Simple check for crashing, needs to be defined based on your simulation parameters
        x, y, vx, vy, angle, angular_velocity, leg_1, leg_2  = self.state
        return y <= 0 and (np.abs(vx) > 0.5 or np.abs(vy) > 0.5 or np.abs(angle) > 0.5)
  
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        # Unpack the state
        x, y, vx, vy, angle, angular_velocity, leg_1, leg_2  = self.state

        # Compute the dynamics based on action inputs
        main_throttle, lateral_throttle = action
        thrust = main_throttle * 0.5  
        lateral_thrust = lateral_throttle * 0.1  # Lateral thrust
        vx += thrust * np.cos(angle) - lateral_thrust * np.sin(angle)
        vy += thrust * np.sin(angle) + lateral_thrust * np.cos(angle) + self.gravity

        # Apply wind effects if enabled
        if self.enable_wind:
            vx += self.wind_power * np.random.randn()
            vy += self.turbulence_power * np.random.randn()

        # Update position
        x += vx
        y += vy

        # Update the state
        self.state = np.array([x, y, vx, vy, angle, angular_velocity,leg_1, leg_2])

        # Compute reward
        target_x, target_y = 0, 0  # Assume the target is at the origin
        distance = np.sqrt((x - target_x)**2 + (y - target_y)**2)
        reward = -distance - (np.abs(vx) + np.abs(vy)) - np.abs(angle) * 10
        # reward += 10 * sum(leg_1 + leg_2)  # 10 points for each leg contact
        reward -= 0.3 if main_throttle > 0 else 0
        reward -= 0.03 if lateral_throttle != 0 else 0

        # Check for landing or crashing
        reward += 100 if self.has_landed() else 0
        reward -= 100 if self.has_crashed() else 0

        # Check termination condition
        done = self.step_counter >= self._max_episode_steps or self.has_crashed() or self.has_landed()
        self.step_counter += 1

        return self.state, reward, done, {}


    # def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
    #     # Unpack the state
    #     x, y, vx, vy, angle, angular_velocity = self.state

    #     # Compute the dynamics
    #     main_throttle, lateral_throttle = action
    #     thrust = main_throttle * 0.5  # Scale thrust between 0 and 0.5
    #     lateral_thrust = lateral_throttle * 0.1  # Scale lateral thrust

    #     # Update the state based on the physics
    #     # Placeholder physics, adjust according to the true dynamics of your environment
    #     vx += thrust * np.cos(angle) - lateral_thrust * np.sin(angle)
    #     vy += thrust * np.sin(angle) + lateral_thrust * np.cos(angle) + self.gravity

    #     # Wind effects (if enabled)
    #     if self.enable_wind:
    #         vx += self.wind_power * np.random.randn()
    #         vy += self.turbulence_power * np.random.randn()

    #     # Update the position
    #     x += vx
    #     y += vy

    #     self.state = np.array([x, y, vx, vy, angle, angular_velocity])

    #     # Compute reward and check termination condition
    #     reward = -np.sqrt(x**2 + y**2)  # Example reward function: negative distance from origin
    #     done = self.step_counter >= self._max_episode_steps
    #     self.step_counter += 1

    #     return self.state, reward, done, {}

    def predict_done(self, state: np.ndarray) -> bool:
        # Unpack the state array
        x, y, vx, vy, angle, angular_velocity, leg_1, leg_2  = state

        # Define landing and crashing conditions
        # Landed safely: close to target area with low velocity and minimal tilt
        landed = y <= 0.1 and leg_1 and leg_2 and abs(vx) < 0.05 and abs(vy) < 0.05 and abs(angle) < 0.1
       
        # Crashed: close to ground but with high velocity or excessive tilt
        crashed = y <= 0.1 and (abs(vx) > 0.1 or abs(vy) > 0.1 or abs(angle) > 0.2)
        # Optionally, check if out of bounds, assuming some max distance from origin
        out_of_bounds = np.abs(x) > 5 or np.abs(y) > 5  # Example bounds

        # Return True if any of the terminal conditions are met
        return landed or crashed or out_of_bounds

    def seed(self, seed: int):
        np.random.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self.init_space.seed(seed)

    def unsafe(self, state: np.ndarray) -> bool:
        # Unpack the state array
        x, y, vx, vy, angle, angular_velocity, leg_1, leg_2 = state

        # Define unsafe conditions
        high_velocity = np.abs(vx) > 1.0 or np.abs(vy) > 1.0
        extreme_angle = np.abs(angle) > 0.25  # More conservative than safe angle
        low_altitude_unsafe = y < 0.1 and (np.abs(vx) > 0.2 or np.abs(angle) > 0.1)
        leg_danger = not (leg_1 or leg_2) and y < 0.1  # Unsafe if no legs are in contact close to ground

        # Combine conditions
        return high_velocity or extreme_angle or low_altitude_unsafe or leg_danger
