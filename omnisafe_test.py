import omnisafe


env_id = 'SafetyPointGoal1-v0'

agent = omnisafe.Agent('CPO', env_id)
# agent.learn()

print(dir(agent.agent))