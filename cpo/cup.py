import torch
import torch.nn.functional as F
from torch.optim import Adam
from cpo.model import CPOActorCritic
import numpy as np
from cpo.utils import RunningMeanStd
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Normal
from torch.nn.utils import clip_grad_norm_

class Lagrange:
    """A simple Lagrange multiplier class for constrained optimization."""
    def __init__(self, initial_value: float, lr: float, cost_limit: float):
        self.cost_limit = cost_limit
        self.lr = lr
        self.lagrangian_multiplier = torch.tensor(initial_value, dtype=torch.float32, requires_grad=False)
    
    def update_lagrange_multiplier(self, current_cost: float):
        """Update the Lagrange multiplier using the current cost."""
        with torch.no_grad():
            self.lagrangian_multiplier += self.lr * (current_cost - self.cost_limit)
            self.lagrangian_multiplier.clamp_(0.0)
    
    def to(self, device: torch.device):
        """Move the multiplier to the specified device."""
        self.lagrangian_multiplier = self.lagrangian_multiplier.to(device)
        return self

class CUP:
    def __init__(self, obs_dim, action_space, args):
        self.gamma = getattr(args, "gamma", 0.99)
        self.cost_gamma = getattr(args, "cost_gamma", 0.99)
        self.lam = getattr(args, "lam", 0.95)
        self.eps_clip = getattr(args, "eps_clip", 0.2)
        self.entropy_coeff = getattr(args, "entropy_coeff", 0.01)
        self.device = torch.device("cuda" if getattr(args, "cuda", True) else "cpu")
        self.max_grad_norm = getattr(args, "max_grad_norm", 0.5)
        print(f"Using device: {self.device}")

        self.trust_region_delta = getattr(args, "cup_trust_region", 0.01)
        self.batch_size = getattr(args, "mini_batch_size", 64)
        self.cost_limit = float(getattr(args, "cost_limit", 10.0))
        
        # Lagrange multiplier
        self.lagrange = Lagrange(
            initial_value=getattr(args, "lagrange_init", 1.0),
            lr=getattr(args, "lagrange_lr", 0.01),
            cost_limit=self.cost_limit
        ).to(self.device)

        # Actor-Critic network
        self.actor_critic = CPOActorCritic(obs_dim, action_space, args.hidden_size).to(self.device)
        self.actor_params = list(self.actor_critic.actor.parameters()) + [self.actor_critic.actor_logstd]
        self.critic_params = list(self.actor_critic.critic.parameters())
        self.cost_critic_params = list(self.actor_critic.cost_critic.parameters())
        
        # Optimizers
        self.actor_optimizer = Adam(self.actor_params, lr=args.actor_lr)
        self.critic_optimizer = Adam(self.critic_params, lr=args.critic_lr)
        self.cost_critic_optimizer = Adam(self.cost_critic_params, lr=args.critic_lr)
        
        # State normalization
        self.state_rms = RunningMeanStd(shape=obs_dim)
        self.debug = getattr(args, "debug", False)
        
        # For KL calculations
        self._p_dist = None

    @torch.no_grad()
    def select_action(self, state):
        state_normalized = np.clip((state - self.state_rms.mean) / (self.state_rms.var**0.5 + 1e-8), -10, 10)
        state_tensor = torch.from_numpy(state_normalized).float().to(self.device).unsqueeze(0)
        action, log_prob = self.actor_critic.act(state_tensor)
        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0]

    @torch.no_grad()
    def process_data(self, memory):
        raw_states = np.array(memory.states[:memory.size])
        raw_rewards = np.array(memory.rewards[:memory.size])
        raw_costs = np.array(memory.costs[:memory.size])
        raw_next_states = np.array(memory.next_states[:memory.size])

        # Update state normalization
        self.state_rms.update(raw_states)
        states = np.clip((raw_states - self.state_rms.mean) / (self.state_rms.var**0.5 + 1e-8), -10, 10)
        next_states = np.clip((raw_next_states - self.state_rms.mean) / (self.state_rms.var**0.5 + 1e-8), -10, 10)
        
        # Convert to tensors
        states_t = torch.from_numpy(states).float().to(self.device)
        rewards_t = torch.from_numpy(raw_rewards).float().to(self.device)
        costs_t = torch.from_numpy(raw_costs).float().to(self.device)
        dones_t = torch.from_numpy(np.array(memory.dones[:memory.size])).float().to(self.device)
        next_states_t = torch.from_numpy(next_states).float().to(self.device)
        actions_t = torch.from_numpy(np.array(memory.actions[:memory.size])).float().to(self.device)

        # Get value estimates
        values = self.actor_critic.get_value(states_t).squeeze()
        next_values = self.actor_critic.get_value(next_states_t).squeeze()
        cost_values = self.actor_critic.get_cost_value(states_t).squeeze()
        next_cost_values = self.actor_critic.get_cost_value(next_states_t).squeeze()

        # Compute returns and advantages
        N = len(rewards_t)
        returns, advantages = torch.zeros(N, device=self.device), torch.zeros(N, device=self.device)
        cost_returns, cost_advantages = torch.zeros(N, device=self.device), torch.zeros(N, device=self.device)
        gae, cost_gae = 0.0, 0.0

        for t in reversed(range(N)):
            mask = 1.0 - dones_t[t]
            
            # Reward advantage
            delta = rewards_t[t] + self.gamma * next_values[t] * mask - values[t]
            gae = delta + self.gamma * self.lam * mask * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
            
            # Cost advantage
            cost_delta = costs_t[t] + self.cost_gamma * next_cost_values[t] * mask - cost_values[t]
            cost_gae = cost_delta + self.cost_gamma * self.lam * mask * cost_gae 
            cost_advantages[t] = cost_gae
            cost_returns[t] = cost_gae + cost_values[t]

        # Old log probabilities
        log_probs_old = self.actor_critic.get_log_prob(states_t, actions_t)
        
        return {
            'states': states_t, 
            'actions': actions_t, 
            'log_probs_old': log_probs_old,
            'returns': returns, 
            'advantages': advantages,
            'cost_returns': cost_returns, 
            'cost_advantages': cost_advantages,
            'values_old': values, 
            'cost_values_old': cost_values,
            'raw_costs': raw_costs,
        }

    def _loss_pi_cost(self, states, actions, logp_old, cost_advantages):
        """CUP cost projection loss."""
        # Get current policy's distribution and log prob
        current_dist = self.actor_critic.get_distribution(states)
        logp_current = current_dist.log_prob(actions).sum(dim=-1)
        ratio = torch.exp(logp_current - logp_old)
        
        # KL divergence with stored old distribution
        kl = torch.distributions.kl_divergence(current_dist, self._p_dist).sum(dim=-1, keepdim=True)
        
        # CUP coefficient
        coef = (1 - self.gamma * self.lam) / (1 - self.gamma)
        
        # Final loss: lambda * coefficient * ratio * cost_advantage + KL
        loss = (self.lagrange.lagrangian_multiplier * coef * ratio * cost_advantages + kl).mean()
        
        return loss

    def _update_ppo_phase(self, data, epochs, batch_size):
        """Standard PPO update for reward maximization."""
        dataset = TensorDataset(
            data['states'], data['actions'], data['log_probs_old'],
            data['advantages'], data['returns'], data['values_old'],
            data['cost_returns'], data['cost_values_old']
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        policy_losses, value_losses, cost_value_losses, entropies, clip_fractions = [], [], [], [], []
        
        for epoch in range(epochs):
            for batch in loader:
                states_b, actions_b, logp_old_b, adv_b, returns_b, values_old_b, cost_returns_b, cost_values_old_b = batch
                
                # Normalize advantages per batch
                adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)
                
                # Get current policy evaluation
                logp, entropy, values, cost_values = self.actor_critic.evaluate(states_b, actions_b)
                
                # PPO policy loss
                ratios = torch.exp(logp - logp_old_b)
                pi_loss1 = ratios * adv_b
                pi_loss2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * adv_b
                policy_loss = -torch.min(pi_loss1, pi_loss2).mean() - self.entropy_coeff * entropy.mean()
                
                # Value loss with clipping
                values_clipped = values_old_b.unsqueeze(1) + torch.clamp(
                    values - values_old_b.unsqueeze(1), -self.eps_clip, self.eps_clip
                )
                value_loss1 = F.mse_loss(values, returns_b.unsqueeze(1))
                value_loss2 = F.mse_loss(values_clipped, returns_b.unsqueeze(1))
                value_loss = torch.max(value_loss1, value_loss2)
                
                # Cost value loss with clipping
                cost_values_clipped = cost_values_old_b.unsqueeze(1) + torch.clamp(
                    cost_values - cost_values_old_b.unsqueeze(1), -self.eps_clip, self.eps_clip
                )
                cost_value_loss1 = F.mse_loss(cost_values, cost_returns_b.unsqueeze(1))
                cost_value_loss2 = F.mse_loss(cost_values_clipped, cost_returns_b.unsqueeze(1))
                cost_value_loss = torch.max(cost_value_loss1, cost_value_loss2)
                
                # Update actor
                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                clip_grad_norm_(self.actor_params, self.max_grad_norm)
                self.actor_optimizer.step()
                
                # Update critics
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                clip_grad_norm_(self.critic_params, self.max_grad_norm)
                self.critic_optimizer.step()
                
                self.cost_critic_optimizer.zero_grad()
                cost_value_loss.backward()
                clip_grad_norm_(self.cost_critic_params, self.max_grad_norm)
                self.cost_critic_optimizer.step()
                
                # Store metrics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                cost_value_losses.append(cost_value_loss.item())
                entropies.append(entropy.mean().item())
                clip_fractions.append((torch.abs(ratios - 1.0) > self.eps_clip).float().mean().item())
        
        return {
            "avg_reward_policy_loss": np.mean(policy_losses),
            "avg_value_loss": np.mean(value_losses),
            "avg_cost_value_loss": np.mean(cost_value_losses),
            "entropy": np.mean(entropies),
            "clip_fraction": np.mean(clip_fractions),
        }

    def _update_cup_phase(self, data, epochs, batch_size):
        """CUP cost projection phase."""
        # Store old distribution before cost updates
        with torch.no_grad():
            old_distribution = self.actor_critic.get_distribution(data['states'])
            old_mean = old_distribution.mean
            old_std = old_distribution.stddev
        
        dataset = TensorDataset(
            data['states'], data['actions'], data['log_probs_old'],
            data['cost_advantages'], old_mean, old_std
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        cost_losses = []
        final_steps = epochs
        
        for epoch in range(epochs):
            epoch_cost_losses = []
            for batch in loader:
                states_b, actions_b, logp_old_b, cost_adv_b, old_mean_b, old_std_b = batch
                
                # Create distribution for this batch
                self._p_dist = Normal(old_mean_b, old_std_b)
                
                # Calculate cost loss
                cost_loss = self._loss_pi_cost(states_b, actions_b, logp_old_b, cost_adv_b)
                epoch_cost_losses.append(cost_loss.item())
                
                # Update actor with cost loss
                self.actor_optimizer.zero_grad()
                cost_loss.backward()
                clip_grad_norm_(self.actor_params, self.max_grad_norm)
                self.actor_optimizer.step()
            
            cost_losses.extend(epoch_cost_losses)
            
            # KL early stopping (like official implementation)
            with torch.no_grad():
                new_distribution = self.actor_critic.get_distribution(data['states'])
                kl = torch.distributions.kl_divergence(new_distribution, old_distribution).mean()
                if self.debug:
                    print(f"CUP Phase - Epoch {epoch+1}: KL = {kl.item():.6f}")
                if kl > self.trust_region_delta:
                    final_steps = epoch + 1
                    if self.debug:
                        print(f"CUP Phase - Early stopping at epoch {epoch+1} due to KL constraint")
                    break
        
        return {
            "avg_cost_policy_loss": np.mean(cost_losses),
            "cup_stop_iter": final_steps,
            "kl_divergence": kl.item() if 'kl' in locals() else 0.0,
        }

    def update_parameters(self, memory, epochs, batch_size):
        """Main update function following official CUP structure."""
        # Process rollout data
        data = self.process_data(memory)
        avg_rollout_cost = np.mean(data['raw_costs'])
        
        # Update Lagrange multiplier first
        self.lagrange.update_lagrange_multiplier(avg_rollout_cost)
        
        if self.debug:
            print(f"Rollout cost: {avg_rollout_cost:.4f}, Lagrange multiplier: {self.lagrange.lagrangian_multiplier.item():.4f}")
            print(f"Advantage stats - mean: {data['advantages'].mean():.6f}, std: {data['advantages'].std():.6f}")
            print(f"Cost advantage stats - mean: {data['cost_advantages'].mean():.6f}, std: {data['cost_advantages'].std():.6f}")
        
        metrics = {}
        
        # Phase 1: Standard PPO update (reward maximization)
        ppo_metrics = self._update_ppo_phase(data, epochs, batch_size)
        metrics.update(ppo_metrics)
        
        # Phase 2: CUP cost projection
        cup_metrics = self._update_cup_phase(data, epochs, batch_size)
        metrics.update(cup_metrics)
        
        # Compute explained variance
        with torch.no_grad():
            final_values = self.actor_critic.get_value(data['states']).squeeze()
            final_cost_values = self.actor_critic.get_cost_value(data['states']).squeeze()
            
            var_y = torch.var(data['returns'])
            explained_var_value = (1 - torch.var(data['returns'] - final_values) / (var_y + 1e-8)).item()
            
            var_y_cost = torch.var(data['cost_returns'])
            explained_var_cost_value = (1 - torch.var(data['cost_returns'] - final_cost_values) / (var_y_cost + 1e-8)).item()
        
        metrics.update({
            "multiplier": self.lagrange.lagrangian_multiplier.item(),
            "explained_var_value": explained_var_value,
            "explained_var_cost_value": explained_var_cost_value,
            "avg_rollout_cost": avg_rollout_cost,
        })
        
        memory.clear_memory()
        return metrics

    def save_checkpoint(self, path):
        """Save model checkpoint."""
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'state_rms_mean': self.state_rms.mean,
            'state_rms_var': self.state_rms.var,
            'state_rms_count': self.state_rms.count,
            'lagrange_multiplier': self.lagrange.lagrangian_multiplier,
        }, path)

    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.state_rms.mean = checkpoint['state_rms_mean']
        self.state_rms.var = checkpoint['state_rms_var']
        self.state_rms.count = checkpoint['state_rms_count']
        self.lagrange.lagrangian_multiplier = checkpoint['lagrange_multiplier'].to(self.device)