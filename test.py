from benchmarks.lunar_lander import LunarLanderEnv
from benchmarks.bipedal_walker import BipedalWalkerEnv
from benchmarks.safety_gym import SafetyPointGoalEnv
from stable_baselines3 import SAC

env = SafetyPointGoalEnv()
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)