# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.distributions import Beta, Normal
# import numpy as np
# import gym
# from collections import deque
# from ppo.utils import *
# # PPO Agent Definition

# class PPOAgent:
#     def __init__(
#         self,
#         state_dim,
#         action_dim,
#         net_width=64,
#         lr=3e-4,
#         gamma=0.99,
#         lam=0.95,
#         clip_epsilon=0.2,
#         update_epochs=10,
#         minibatch_size=64,
#         entropy_coef=0.0,
#         value_coef=0.5,
#         max_grad_norm=0.5,
#         use_gpu=False
#     ):
#         self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')

#         # Initialize the actor and critic networks
#         self.actor = GaussianActor_mu(state_dim, action_dim, net_width).to(self.device)
#         self.critic = Critic(state_dim, net_width).to(self.device)

#         # Optimizers
#         self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
#         self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

#         # Hyperparameters
#         self.gamma = gamma
#         self.lam = lam
#         self.clip_epsilon = clip_epsilon
#         self.update_epochs = update_epochs
#         self.minibatch_size = minibatch_size
#         self.entropy_coef = entropy_coef
#         self.value_coef = value_coef
#         self.max_grad_norm = max_grad_norm

#         # Storage for training
#         self.states = []
#         self.actions = []
#         self.log_probs = []
#         self.rewards = []
#         self.masks = []
#         self.values = []

#     def select_action(self, state):
#         state_tensor = torch.FloatTensor(state).to(self.device)
#         dist = self.actor.get_dist(state_tensor)
#         value = self.critic(state_tensor)
#         action = dist.sample()
#         action = action.clamp(-1.0, 1.0)  # Assuming action space is between -1 and 1
#         log_prob = dist.log_prob(action).sum(dim=-1)
#         return action.cpu().detach().numpy(), log_prob.cpu().detach().numpy(), value.cpu().detach().numpy()

#     def store_transition(self, state, action, reward, log_prob, value, mask):
#         self.states.append(state)
#         self.actions.append(action)
#         self.rewards.append(reward)
#         self.log_probs.append(log_prob)
#         self.values.append(value)
#         self.masks.append(mask)

#     def compute_returns_and_advantages(self, next_value):
#         rewards = []
#         gae = 0
#         values = self.values + [next_value]
#         for step in reversed(range(len(self.rewards))):
#             delta = self.rewards[step] + self.gamma * values[step + 1] * self.masks[step] - values[step]
#             gae = delta + self.gamma * self.lam * self.masks[step] * gae
#             rewards.insert(0, gae + values[step])
#         return rewards

#     def update(self):
#         # Convert buffers to tensors
#         states = torch.FloatTensor(self.states).to(self.device)
#         actions = torch.FloatTensor(self.actions).to(self.device)
#         old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
#         returns = torch.FloatTensor(self.compute_returns_and_advantages(next_value=0)).to(self.device)
#         advantages = returns - torch.FloatTensor(self.values).to(self.device)

#         # Normalize advantages
#         advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

#         # PPO Update
#         for _ in range(self.update_epochs):
#             for idx in self.mini_batch_generator(len(self.states)):
#                 sampled_states = states[idx]
#                 sampled_actions = actions[idx]
#                 sampled_old_log_probs = old_log_probs[idx]
#                 sampled_returns = returns[idx]
#                 sampled_advantages = advantages[idx]

#                 dist = self.actor.get_dist(sampled_states)
#                 entropy = dist.entropy().mean()
#                 new_log_probs = dist.log_prob(sampled_actions).sum(dim=-1)
#                 ratio = torch.exp(new_log_probs - sampled_old_log_probs)

#                 # Surrogate loss
#                 surr1 = ratio * sampled_advantages
#                 surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * sampled_advantages
#                 actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

#                 # Value function loss
#                 value = self.critic(sampled_states).squeeze()
#                 critic_loss = self.value_coef * F.mse_loss(value, sampled_returns)

#                 # Update actor
#                 self.actor_optimizer.zero_grad()
#                 actor_loss.backward()
#                 nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
#                 self.actor_optimizer.step()

#                 # Update critic
#                 self.critic_optimizer.zero_grad()
#                 critic_loss.backward()
#                 nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
#                 self.critic_optimizer.step()

#         # Clear storage
#         self.states = []
#         self.actions = []
#         self.log_probs = []
#         self.rewards = []
#         self.masks = []
#         self.values = []

#     def mini_batch_generator(self, batch_size):
#         indices = np.arange(batch_size)
#         np.random.shuffle(indices)
#         for start_idx in range(0, batch_size, self.minibatch_size):
#             end_idx = start_idx + self.minibatch_size
#             yield indices[start_idx:end_idx]

#     def train(self, env, total_timesteps, log_interval=10):
#         timestep = 0
#         episode_rewards = deque(maxlen=10)
#         state = env.reset()
#         episode_reward = 0
#         episode_length = 0

#         while timestep < total_timesteps:
#             action, log_prob, value = self.select_action(state)
#             next_state, reward, done, _ = env.step(action)
#             mask = 0 if done else 1
#             self.store_transition(state, action, reward, log_prob, value, mask)

#             state = next_state
#             episode_reward += reward
#             episode_length += 1
#             timestep += 1

#             if done:
#                 state = env.reset()
#                 episode_rewards.append(episode_reward)
#                 episode_reward = 0
#                 episode_length = 0

#             # Update agent after collecting a batch of data
#             if len(self.states) >= self.minibatch_size:
#                 with torch.no_grad():
#                     next_state_tensor = torch.FloatTensor(next_state).to(self.device)
#                     next_value = self.critic(next_state_tensor).cpu().numpy()
#                 self.update()

#             # Logging
#             if timestep % (log_interval * self.minibatch_size) == 0:
#                 avg_reward = np.mean(episode_rewards) if episode_rewards else 0
#                 print(f"Time step: {timestep}, Average Reward: {avg_reward:.2f}")

# # Training Loop Example

# if __name__ == "__main__":
#     # Create environment
#     env = gym.make('Pendulum-v1')  # Continuous action space between -1 and 1
#     state_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.shape[0]

#     # Initialize agent
#     agent = PPOAgent(
#         state_dim=state_dim,
#         action_dim=action_dim,
#         net_width=64,
#         lr=3e-4,
#         gamma=0.99,
#         lam=0.95,
#         clip_epsilon=0.2,
#         update_epochs=10,
#         minibatch_size=64,
#         entropy_coef=0.0,
#         value_coef=0.5,
#         max_grad_norm=0.5,
#         use_gpu=False  # Set to True if you have a CUDA-compatible GPU
#     )

#     # Train the agent
#     total_timesteps = 1_000_000
#     agent.train(env, total_timesteps)