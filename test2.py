from benchmarks.lunar_lander import LunarLanderEnv
from benchmarks.bipedal_walker import BipedalWalkerEnv
from gymnasium.envs.box2d.lunar_lander import step_api_compatibility, heuristic
from gymnasium.envs.box2d.bipedal_walker import BipedalWalkerHeuristics
import gymnasium as gym
import numpy as np

env = BipedalWalkerEnv()
s, info = env.reset()
steps = 0
total_reward = 0
states = [s]
a = np.array([0.0, 0.0, 0.0, 0.0])
# Heurisic: suboptimal, have no notion of balance.
heuristics = BipedalWalkerHeuristics()
while True:
    s, r, terminated, truncated, info = env.step(a)
    
    total_reward += r
    if steps % 20 == 0 or terminated or truncated:
        print("\naction " + str([f"{x:+0.2f}" for x in a]))
        print(f"step {steps} total_reward {total_reward:+0.2f}")
        print("hull " + str([f"{x:+0.2f}" for x in s[0:4]]))
        print("leg0 " + str([f"{x:+0.2f}" for x in s[4:9]]))
        print("leg1 " + str([f"{x:+0.2f}" for x in s[9:14]]))
    steps += 1

    a = heuristics.step_heuristic(s)
    states.append(s)

    if terminated or truncated:
        break
    
    

print(total_reward)

print(np.min(np.array(states[:-50]), axis=0))
print(np.max(np.array(states[:-50]), axis=0))