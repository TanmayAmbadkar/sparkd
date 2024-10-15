import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Actor-Critic Model
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorCritic, self).__init__()
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value


# PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim, hidden_dim = 64, clip_param=0.2, lr=3e-4, gamma=0.99, gae_lambda=0.95, epochs=10, batch_size=128):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.epochs = epochs
        self.batch_size = batch_size

        self.model = ActorCritic(state_dim, action_dim, hidden_dim=hidden_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.old_model = ActorCritic(state_dim, action_dim, hidden_dim=hidden_dim)
        self.old_model.load_state_dict(self.model.state_dict())

    def action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs, _ = self.model(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action).item()
        return action.item(), log_prob
    
    def select_action(self, state):
        action, log_prob = self.action(state)
        return action


    def compute_advantages(self, rewards, values, dones):
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        return advantages

    def optimize(self, memory):
        states, actions, rewards, next_states, dones = memory.sample(len(memory))
        
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        old_log_probs = torch.FloatTensor(log_probs)

        _, values = self.old_model(states)
        _, next_values = self.old_model(next_states)
        values = values.squeeze()
        next_values = next_values.squeeze()

        advantages = self.compute_advantages(rewards, values.detach().numpy(), dones)
        advantages = torch.FloatTensor(advantages)

        returns = advantages + values

        for _ in range(self.epochs):
            for batch_start in range(0, len(states), self.batch_size):
                batch_end = batch_start + self.batch_size
                batch_indices = slice(batch_start, batch_end)
                
                # Get batches
                state_batch = states[batch_indices]
                action_batch = actions[batch_indices]
                return_batch = returns[batch_indices]
                advantage_batch = advantages[batch_indices].detach()
                old_log_prob_batch = old_log_probs[batch_indices].detach()

                # Calculate current policy and value
                action_probs, values = self.model(state_batch)
                dist = torch.distributions.Categorical(action_probs)
                log_probs = dist.log_prob(action_batch)

                # Calculate ratio
                ratios = torch.exp(log_probs - old_log_prob_batch)

                # Clip the ratio to avoid large policy updates
                surr1 = ratios * advantage_batch
                surr2 = torch.clamp(ratios, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage_batch
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = (return_batch - values.squeeze()).pow(2).mean()

                # Total loss
                loss = policy_loss + 0.5 * value_loss

                # Update the model
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # Update the old model after training
        self.old_model.load_state_dict(self.model.state_dict())

