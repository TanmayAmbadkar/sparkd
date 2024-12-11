from benchmarks.safety_gym import SafetyPointGoalEnv

env = SafetyPointGoalEnv()

state, info = env.reset()    

while True:
    
    action = env.action_space.sample()
    state, reward, done, trunc, info = env.step(action)
    
    if env.unsafe(state):
        print("UNSAFE")
        break
    elif done or trunc:
        state, info = env.reset()    