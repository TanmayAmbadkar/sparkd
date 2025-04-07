from pcc.pcc_predictor import PCCPredictor, fit_pcc
import gymnasium as gym
import torch
import numpy as np

env = gym.make("LunarLander-v3", continuous=True)
x_dim = env.observation_space.shape[0]
z_dim = 4
u_dim = env.action_space.shape[0]

amortized = True
lr = 1e-3
weight_decay = 1e-4
states, actions, next_states = [], [], []

# Collect data
while len(states) < 10000:
    state, info = env.reset()
    done = False
    truncated = False
    while not done and not truncated:
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        states.append(state)
        actions.append(action)
        next_states.append(next_state)
        state = next_state


model = PCCPredictor(x_dim, z_dim, u_dim, amortized=amortized, lr=lr, weight_decay=weight_decay)
fit_pcc(states, actions, next_states, model,
        epochs=50, batch_size=128, lr=lr, weight_decay=weight_decay,
        amortized=amortized
    )

print(model.model.predict(states[:15], actions[:15]))

# Test the PCC model
env = gym.make("LunarLander-v3", continuous=True)
state, info = env.reset()
done = False
truncated = False
while not done and not truncated:
    action = env.action_space.sample()
    next_state, reward, done, truncated, info = env.step(action)
    # Test the model
    # Use the model to predict the next state
    _, x_next = model.model.predict(state.reshape(1, -1), action.reshape(1, -1))
    print(f"Next State: {np.round(next_state, 2)}\npredicted Next State: {np.round(x_next.numpy().reshape(-1, ), 2)}")
    state = next_state
