import gymnasium

from pcc.pcc_predictor import fit_pcc
env = gymnasium.make("CartPole-v1")

# Example data generation
states = []
actions = []
next_states = []
state, _= env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    next_state, rew, done, trunc, info = env.step(action)
    states.append(state)
    actions.append(action)
    next_states.append(next_state)

    if done or trunc:        
        state, _= env.reset()

model = fit_pcc(states, actions, next_states, x_dim=10, z_dim=5, u_dim=5)
print("Model training complete.")