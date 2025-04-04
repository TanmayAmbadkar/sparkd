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
        epochs=30, batch_size=128, lr=lr, weight_decay=weight_decay,
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
    print(f"Next State: {np.round(next_state, 3)}\npredicted Next State: {np.round(x_next.numpy().reshape(-1, ), 3)}")
    state = next_state

        

        
# import gymnasium as gym
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error
# import torch


# # -----------------------------
# # Step 1. Data Collection
# # -----------------------------
# def collect_observations(num_episodes=100):
#     """
#     Run the LunarLander environment and collect state observations.
#     Each state is an 8-dimensional vector.
#     """
#     env = gym.make("LunarLander-v3")
#     observations = []
#     for ep in range(num_episodes):
#         # For gymnasium, reset returns (observation, info)
#         state, _ = env.reset()
#         done = False
#         while not done:
#             # Sample a random action
#             action = env.action_space.sample()
#             # Step returns (observation, reward, done, truncated, info)
#             next_state, reward, done, truncated, info = env.step(action)
#             observations.append(state)
#             state = next_state
#             # Optionally, you can break on truncation or use it as a done flag.
#             if truncated:
#                 break
#     env.close()
#     return np.array(observations)

# # -----------------------------
# # Step 2. Define the VAE Model
# # -----------------------------
# class VAE(nn.Module):
#     def __init__(self, input_dim=8, hidden_dim=64, latent_dim=4):
#         """
#         A simple VAE with one hidden layer in both the encoder and decoder.
#         """
#         super(VAE, self).__init__()
#         # Encoder layers
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc_mu = nn.Linear(hidden_dim, latent_dim)      # Mean of the latent distribution
#         self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # Log-variance of the latent distribution
        
#         # Decoder layers
#         self.fc3 = nn.Linear(latent_dim, hidden_dim)
#         self.fc4 = nn.Linear(hidden_dim, input_dim)
    
#     def encode(self, x):
#         h = torch.relu(self.fc1(x))
#         return self.fc_mu(h), self.fc_logvar(h)
    
#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std
    
#     def decode(self, z):
#         h = torch.relu(self.fc3(z))
#         return self.fc4(h)
    
#     def forward(self, x):
#         mu, logvar = self.encode(x)
#         z = self.reparameterize(mu, logvar)
#         recon = self.decode(z)
#         return recon, mu, logvar

# def vae_loss(recon_x, x, mu, logvar):
#     """
#     VAE loss: reconstruction loss (MSE) plus KL divergence.
#     """
#     recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
#     kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     return recon_loss + kl_loss

# # -----------------------------
# # Step 3. Training Function
# # -----------------------------
# def train_vae(model, dataloader, epochs=50, learning_rate=0.0005):
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     model.train()
#     for epoch in range(epochs):
#         total_loss = 0
#         for batch in dataloader:
#             # batch is a tuple; unpack the tensor
#             batch = batch[0]
#             optimizer.zero_grad()
#             recon_batch, mu, logvar = model(batch)
#             loss = vae_loss(recon_batch, batch, mu, logvar)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         avg_loss = total_loss / len(dataloader.dataset)
#         print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# def test_vae(model, num_episodes=20):
#     """
#     Collects test observations from the Lunar Lander environment, computes the VAE reconstructions,
#     and prints the explained variance score, MSE, and MAE.
    
#     Args:
#         model (torch.nn.Module): The trained VAE model.
#         num_episodes (int): Number of episodes to gather test observations.
        
#     Returns:
#         tuple: (explained_variance, mse, mae)
#     """
#     # Collect new test observations using the same function as in training.
#     test_obs = collect_observations(num_episodes=num_episodes)
#     print("Test observations shape:", test_obs.shape)
    
#     # Convert the observations to a torch tensor
#     test_obs_tensor = torch.tensor(test_obs, dtype=torch.float32)
    
#     # Set model to evaluation mode and get reconstructions without tracking gradients
#     model.eval()
#     with torch.no_grad():
#         recon_test, _, _ = model(test_obs_tensor)
    
#     # Convert tensors to numpy arrays for metric computation
#     test_obs_np = test_obs_tensor.numpy()
#     recon_test_np = recon_test.numpy()
    
#     # Compute the metrics
#     ev_score = explained_variance_score(test_obs_np, recon_test_np)
#     mse_score = mean_squared_error(test_obs_np, recon_test_np)
#     mae_score = mean_absolute_error(test_obs_np, recon_test_np)
    
#     print(f"Explained Variance Score: {ev_score:.4f}")
#     print(f"MSE: {mse_score:.4f}")
#     print(f"MAE: {mae_score:.4f}")
    
#     return ev_score, mse_score, mae_score

# # Example usage:
# # Assuming vae_model is your trained VAE from the previous script.
# # ev, mse, mae = test_vae(vae_model, num_episodes=20)

# # -----------------------------
# # Main Execution
# # -----------------------------
# if __name__ == '__main__':
#     # Collect observations from the environment
#     observations = collect_observations(num_episodes=100)
#     print("Collected observations shape:", observations.shape)

#     # Convert observations to a PyTorch tensor
#     observations_tensor = torch.tensor(observations, dtype=torch.float32)
    
#     # Create a DataLoader for training
#     dataset = TensorDataset(observations_tensor)
#     dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
#     # Initialize the VAE model
#     vae_model = VAE(input_dim=observations.shape[1], hidden_dim=64, latent_dim=2)
    
#     # Train the VAE
#     train_vae(vae_model, dataloader, epochs=50, learning_rate=1e-3)

#     test_vae(vae_model, num_episodes=20)
