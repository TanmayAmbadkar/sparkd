import omnisafe


env_id = 'SafetyPointGoal1-v0'

agent = omnisafe.Agent('CPO', env_id).learn()

# print(dir(agent.agent))