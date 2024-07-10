import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any

class LunarLanderEnv(gym.Env):
    def __init__(self, gravity=-0.25, enable_wind=False, wind_power=0.5, turbulence_power=0.5):
        super().__init__()

        # Constants
        self.gravity = gravity
        self.enable_wind = enable_wind
        self.wind_power = wind_power
        self.turbulence_power = turbulence_power

        # Action space: Continuous, Main engine and lateral boosters
        self.action_space = gym.spaces.Box(low=np.array([0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

        # Observation space: x, y, vx, vy, angle, angular velocity

        # Might need to change
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

        # Initial conditions space: x, y, vx, vy, angle, angular velocity, leg 1, leg 2
        # Getting errors when trying to force leg_1 and leg_2 to be 0, 
        self.init_space = gym.spaces.Box(low=np.array([-0.5, 7.5, -0.1, -0.1, -0.1, -0.1,0,0]), 
                                         high=np.array([0.5, 10, 0.1, 0.1, 0.1, 0.1,0.0000001,0.0000001]), dtype=np.float32)

        # Simulation parameters
        self._max_episode_steps = 400
        self.step_counter = 0


    # def fix_legs(self, state: np.ndarray) -> np.ndarray:
    #     x, y, vx, vy, angle, angular_velocity, leg_1, leg_2 = state
    #     leg_1 = 0 if leg_1 != 0 else 0 
    #     leg_2 = 0 if leg_2 != 0 else 0
    #     return np.array([x, y, vx, vy, angle, angular_velocity, leg_1, leg_2])

    def reset(self) -> np.ndarray:
        
        self.state = self.init_space.sample()

        # Ensure legs start as 0 (bugged)
        # self.state[6] = 0  
        # self.state[7] = 0  

        self.step_counter = 0
        # self.state = self.fix_legs(self.state)

       
        print(f"Initial state: {self.state}")  # debugging

        # retrun f(s)
        return self.state


    def has_landed(self):
        # Simple check for landing
        _, y, vx, vy, angle, _, leg_1, leg_2 = self.state
        return y <= 0.1 and leg_1 and leg_2 and abs(vx) < 0.05 and abs(vy) < 0.05 and abs(angle) < 0.1

    def has_crashed(self):
        # Simple check for crashing
        _, y, vx, vy, angle, _, _, _ = self.state
        return y <= 0.1 and (abs(vx) > 0.5 or abs(vy) > 0.5 or abs(angle) > 0.5)
    
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        x, y, vx, vy, angle, angular_velocity, leg_1, leg_2  = self.state

        # Compute the dynamics based on action inputs
        main_throttle, lateral_throttle = action
        thrust = main_throttle * 2 
        lateral_thrust = lateral_throttle * 0.4  
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


        # Not original
        reward -= 10 if abs(x) > 4 else 0
        reward -= 10 if abs (vx) or abs(vy) > 3.5 else 0
        reward += 10 if abs (vx) or abs (vy) < 1.5 else 0



        # Check termination condition
        if self.has_crashed():
            print("the ship has crashed")
            print("y", y, "vx", vx, "vy ", vy, "angle", angle)

        if self.has_landed():
            print("the ship has landed")

        done = self.step_counter >= self._max_episode_steps or self.has_crashed() or self.has_landed()
        self.step_counter += 1

        #return f(s)


        return self.state, reward, done, {}

    

    def predict_done(self, state: np.ndarray) -> bool:
        # Unpack the state array
        x, y, vx, vy, angle, _, leg_1, leg_2  = state

        # Landing and crashing conditions

        # Landed safely: close to target area with low velocity and minimal tilt
        landed = y <= 0.1 and leg_1 and leg_2 and abs(vx) < 0.05 and abs(vy) < 0.05 and abs(angle) < 0.1
       
        # Crashed: close to ground but with high velocity or excessive tilt
        crashed = y <= 0.1 and (abs(vx) > 0.1 or abs(vy) > 0.1 or abs(angle) > 0.2)

        # Check if out of bounds, assuming some max distance from origin
        out_of_bounds = np.abs(x) > 5 or np.abs(y) > 5  # Example bounds

        # Return True if any of the terminal conditions are met
        return landed or crashed or out_of_bounds

    def seed(self, seed: int):
        np.random.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self.init_space.seed(seed)
    
    
    # Needs to get changed
    def unsafe(self, state: np.ndarray) -> bool:
        # Unpack the state array
        _, y, vx, vy, angle, _, _, _ = state
    
        #  Unsafe conditions
        high_velocity_x = np.abs(vx) > 4.5
        high_velocity_y =  np.abs(vy) > 4.5
    
        extreme_angle = np.abs(angle) > 0.25
        low_altitude_unsafe = y < 0.1 and (np.abs(vx) > 0.2 or np.abs(angle) > 0.1)
        # leg_danger = not (leg_1 or leg_2) and y < 0.1  # Unsafe if no legs are in contact close to ground
        print("Too fast sideways: ", high_velocity_x, "\n", "Too fast down: ", high_velocity_x, "\n", "Too much angle: ", extreme_angle, "\n", "Too low and fast: ", low_altitude_unsafe, "\n")
        print(state)
        # Combine conditions
        return high_velocity_x or high_velocity_y or extreme_angle or low_altitude_unsafe #or leg_danger

