import gymnasium as gym
import numpy as np

class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = env.observation_space.shape
        # Assume (H, W, C) -> (C, H, W)
        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low.transpose(2, 0, 1),
            high=env.observation_space.high.transpose(2, 0, 1),
            shape=(obs_shape[2], obs_shape[0], obs_shape[1]),
            dtype=env.observation_space.dtype
        )

    def observation(self, observation):
        # Transpose (H, W, C) -> (C, H, W)
        return np.transpose(observation, (2, 0, 1))

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
                 env = TransposeImage(env)
        return env

    # Try Safety Gymnasium
    try:
        import safety_gymnasium
        try:
            env = safety_gymnasium.make(name, render_mode=render_mode)
            print(f"Loading {name} from safety_gymnasium.")
            if env is not None:
                 shape = env.observation_space.shape
                 # Heuristic: 3D shape, last dim is channels (1, 3, 4), first dim is spatial (> 5)
                 if len(shape) == 3 and shape[-1] in [1, 3, 4] and shape[0] > 5:
                     print(f"Auto-wrapping environment to format (C, H, W). Original: {shape}")
                     env = TransposeImage(env)
            return env
        except Exception:
            pass
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
            env = TransposeImage(env)
            
    return env
