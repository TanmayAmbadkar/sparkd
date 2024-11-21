def get_env_from_name(name):

    if name == 'acc':
        from .acc import AccEnv
        return AccEnv()
    if name == 'car_racing':
        from .car_racing import CarRacingEnv
        return CarRacingEnv()
    if name == 'mid_obstacle':
        from .mid_obstacle import MidObstacleEnv
        return MidObstacleEnv()
    if name == 'mountain_car':
        from .mountain_car import MountainCarEnv
        return MountainCarEnv()
    if name == 'noisy_road':
        from .noisy_road import NoisyRoadEnv
        return NoisyRoadEnv()
    if name == 'noisy_road_2d':
        from .noisy_road_2d import NoisyRoad2dEnv
        return NoisyRoad2dEnv()
    if name == 'obstacle':
        from .obstacle import ObstacleEnv
        return ObstacleEnv()
    if name == 'pendulum':
        from .pendulum import PendulumEnv
        return PendulumEnv()
    if name == 'road':
        from .road import RoadEnv
        return RoadEnv()
    if name == 'road_2d':
        from .road_2d import Road2dEnv
        return Road2dEnv()
    if name == 'lunar_lander':
        from .lunar_lander import LunarLanderEnv
        return LunarLanderEnv()
    if name == 'lunar_lander_R':
        from .lunar_lander_RedDim import LunarLanderEnv2
        return LunarLanderEnv2()
    if name == 'bipedal_walker':
        from .bipedal_walker import BipedalWalkerEnv
        return BipedalWalkerEnv()
    if name == 'inverted_pendulum':
        from .inverted_pendulum import InvertedDoublePendulumEnv
        return InvertedDoublePendulumEnv()
    if name == 'hopper':
        from .hopper import HopperEnv
        return HopperEnv()
    if name == 'safety_point':
        from .safety_gym import SafetyPointGoalEnv
        return SafetyPointGoalEnv()
    if name == 'carplatoon4':
        from .CarPlatoon4 import CarPlatoonEnv
        return CarPlatoonEnv()
    if name == 'Oscillator':
        from .Oscillator import OscillatorEnv
        return OscillatorEnv()
    else:
        raise RuntimeError("Unkonwn environment: " + name)
