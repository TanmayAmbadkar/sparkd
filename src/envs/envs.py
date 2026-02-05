import gymnasium as gym
import numpy as np
import traceback

class TransposeImage:
    def __init__(self, env):
        self.env = env
        obs_shape = env.observation_space.shape
        # Assume (H, W, C) -> (C, H, W)
        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low.transpose(2, 0, 1),
            high=env.observation_space.high.transpose(2, 0, 1),
            shape=(obs_shape[2], obs_shape[0], obs_shape[1]),
            dtype=env.observation_space.dtype
        )
        self.action_space = env.action_space

    def __getattr__(self, name):
        return getattr(self.env, name)

    def observation(self, observation):
        # Transpose (H, W, C) -> (C, H, W)
        return np.transpose(observation, (2, 0, 1))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def step(self, action):
        ret = self.env.step(action)
        # Handle 6-tuple (Safety Gym)
        if len(ret) == 6:
            obs, reward, cost, terminated, truncated, info = ret
            return self.observation(obs), reward, cost, terminated, truncated, info
        # Handle 5-tuple (Standard Gym)
        elif len(ret) == 5:
            obs, reward, terminated, truncated, info = ret
            return self.observation(obs), reward, terminated, truncated, info
        else:
            raise ValueError(f"Unexpected step return length: {len(ret)}")
            
    def render(self):
        return self.env.render()

def get_env_from_name(name, render_mode=None):
    env = None
    
class SafetyVisionWrapper:
    def __init__(self, env):
        self.env = env
        # Verify it is a dict space with vision
        if isinstance(env.observation_space, gym.spaces.Dict) and 'vision' in env.observation_space.spaces:
            self.observation_space = env.observation_space['vision']
        else:
            self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.last_frame = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if isinstance(obs, dict) and 'vision' in obs:
            self.last_frame = obs['vision']
            return obs['vision'], info
        self.last_frame = obs # If not dict/vision, best guess
        return obs, info

    def step(self, action):
        # Support both 5-tuple (Gym) and 6-tuple (Safety Gym) returns
        ret = self.env.step(action)
        if len(ret) == 6:
            obs, reward, cost, terminated, truncated, info = ret
            if isinstance(obs, dict) and 'vision' in obs:
                self.last_frame = obs['vision']
                obs = obs['vision']
            else:
                 self.last_frame = obs
            return obs, reward, cost, terminated, truncated, info
        elif len(ret) == 5:
            # Standard Gym: obs, reward, terminated, truncated, info
            obs, reward, terminated, truncated, info = ret
            if isinstance(obs, dict) and 'vision' in obs:
                self.last_frame = obs['vision']
                obs = obs['vision']
            else:
                 self.last_frame = obs
            return obs, reward, terminated, truncated, info
        else:
             raise ValueError(f"Unexpected step return length: {len(ret)}")

    def render(self):
        # Safety Gymnasium vision envs forbid render() calls, so we use the cached frame
        if self.last_frame is not None:
             return self.last_frame
        return self.env.render()

class ResizeImage:
    def __init__(self, env, size=(64, 64)):
        self.env = env
        self.size = size
        obs_shape = env.observation_space.shape
        # Update observation space to new size, preserving channels
        # Input assumed to be (H, W, C)
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(size[0], size[1], obs_shape[2]),
            dtype=np.uint8
        )
        self.action_space = env.action_space

    def __getattr__(self, name):
        return getattr(self.env, name)

    def observation(self, observation):
        import cv2
        # Resize inputs (H, W, C) -> (64, 64, C)
        return cv2.resize(observation, (self.size[1], self.size[0]), interpolation=cv2.INTER_AREA)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def step(self, action):
        ret = self.env.step(action)
        if len(ret) == 6:
            obs, reward, cost, terminated, truncated, info = ret
            return self.observation(obs), reward, cost, terminated, truncated, info
        elif len(ret) == 5:
            obs, reward, terminated, truncated, info = ret
            return self.observation(obs), reward, terminated, truncated, info
        else:
             raise ValueError(f"Unexpected step return length: {len(ret)}")
    
    def render(self):
        return self.env.render()

def get_env_from_name(name, render_mode=None):
    env = None
    # Check if exact string match exists in gymnasium
    if name in gym.envs.registry:
        print(f"Loading {name} from gymnasium registry.")
        env = gym.make(name, render_mode=render_mode)
        if env is not None:
             shape = env.observation_space.shape
             # Heuristic: 3D shape, last dim is channels (1, 3, 4), first dim is spatial (> 5)
             if len(shape) == 3 and shape[-1] in [1, 3, 4] and shape[0] > 5:
                 print(f"Auto-wrapping environment to format (C, H, W). Original: {shape}")
                 # Resize first if large
                 if shape[0] > 64:
                     print(f"Resizing from {shape} to (64, 64, {shape[2]})")
                     env = ResizeImage(env)
                 env = TransposeImage(env)
        return env

    # Try Safety Gymnasium
    try:
        import safety_gymnasium
        try:
            # print("safety_gymnasium")
            env = safety_gymnasium.make(name, render_mode=render_mode)
            print(f"Loading {name} from safety_gymnasium.")
            if env is not None:
                 # Check for Dict observation (Vision)
                 if isinstance(env.observation_space, gym.spaces.Dict) and 'vision' in env.observation_space.spaces:
                     print("Detected Dict observation with 'vision' key. Wrapping with SafetyVisionWrapper.")
                     env = SafetyVisionWrapper(env)
                 
                 # Now check for Image Transpose
                 shape = env.observation_space.shape
                 # Heuristic: 3D shape, last dim is channels (1, 3, 4), first dim is spatial (> 5)
                 if len(shape) == 3 and shape[-1] in [1, 3, 4] and shape[0] > 5:
                     print(f"Auto-wrapping environment to format (C, H, W). Original: {shape}")
                     # Resize first if large
                     if shape[0] > 64:
                         print(f"Resizing from {shape} to (64, 64, {shape[2]})")
                         env = ResizeImage(env)
                     env = TransposeImage(env)
            return env
        except Exception:
            print(traceback.format_exc())
    except ImportError:
        pass
    
    if name == 'acc':
        from .acc import AccEnv
        env = AccEnv()
    elif name == 'car_racing':
        from .car_racing import CarRacingEnv
        env = CarRacingEnv()
    elif name == 'mid_obstacle':
        from .mid_obstacle import MidObstacleEnv
        env = MidObstacleEnv()
    elif name == 'mountain_car':
        from .mountain_car import MountainCarEnv
        env = MountainCarEnv()
    elif name == 'noisy_road':
        from .noisy_road import NoisyRoadEnv
        env = NoisyRoadEnv()
    elif name == 'noisy_road_2d':
        from .noisy_road_2d import NoisyRoad2dEnv
        env = NoisyRoad2dEnv()
    elif name == 'obstacle':
        from .obstacle import ObstacleEnv
        env = ObstacleEnv()
    elif name == 'pendulum':
        from .pendulum import PendulumEnv
        env = PendulumEnv()
    elif name == 'road':
        from .road import RoadEnv
        env = RoadEnv()
    elif name == 'road_2d':
        from .road_2d import Road2dEnv
        env = Road2dEnv()
    elif name == 'lunar_lander':
        from .lunar_lander import LunarLanderEnv
        env = LunarLanderEnv(render_mode=render_mode)
    elif name == 'lunar_lander_R':
        from .lunar_lander_RedDim import LunarLanderEnv2
        env = LunarLanderEnv2()
    elif name == 'bipedal_walker':
        from .bipedal_walker import BipedalWalkerEnv
        env = BipedalWalkerEnv()
    elif name == 'inverted_pendulum':
        from .inverted_pendulum import InvertedPendulumEnv
        env = InvertedPendulumEnv()
    elif name == 'hopper':
        from .hopper import HopperEnv
        env = HopperEnv()
    elif name == 'walker':
        from .walker import WalkerEnv
        env = WalkerEnv()
    elif name == 'cheetah':
        from .cheetah import CheetahEnv
        env = CheetahEnv()
    elif name == 'safety_point':
        from .safety_gym import SafetyPointGoalEnv
        env = SafetyPointGoalEnv()
    elif name == 'carplatoon4':
        from .CarPlatoon4 import CarPlatoonEnv
        env = CarPlatoonEnv()
    elif name == 'Oscillator':
        from .Oscillator import OscillatorEnv
        env = OscillatorEnv()
    elif name == 'humanoid':
        from .humanoid import HumanoidEnv
        env = HumanoidEnv()
    elif name == 'ant':
        from .ant import AntEnv
        env = AntEnv(render_mode=render_mode)
    else:
        raise RuntimeError("Unkonwn environment: " + name)
        
    # Auto-wrap channel-last images
    if env is not None:
        shape = env.observation_space.shape
        # Heuristic: 3D shape, last dim is channels (1, 3, 4), first dim is spatial (> 5)
        if len(shape) == 3 and shape[-1] in [1, 3, 4] and shape[0] > 5:
            print(f"Auto-wrapping environment to format (C, H, W). Original: {shape}")
            # Resize first
            if shape[0] > 64:
                 print(f"Resizing from {shape} to (64, 64, {shape[2]})")
                 env = ResizeImage(env)
            env = TransposeImage(env)
            
    return env
