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
        
        self.lagrange = Lagrange(
            initial_value=getattr(args, "lagrange_init", 1.0),
            lr=getattr(args, "lagrange_lr", 0.01),
            cost_limit=self.cost_limit
        ).to(self.device)

        self.actor_critic = CPOActorCritic(obs_dim, action_space, args.hidden_size).to(self.device)
        self.actor_params = list(self.actor_critic.actor.parameters()) + [self.actor_critic.actor_logstd]
        self.critic_params = list(self.actor_critic.critic.parameters())
        self.cost_critic_params = list(self.actor_critic.cost_critic.parameters())
        self.actor_optimizer = Adam(self.actor_params, lr=args.actor_lr)
        self.critic_optimizer = Adam(self.critic_params, lr=args.critic_lr)
        self.cost_critic_optimizer = Adam(self.cost_critic_params, lr=args.critic_lr)
        self.debug = getattr(args, "debug", False)
        
        self.state_rms = RunningMeanStd(shape=obs_dim)

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

        self.state_rms.update(raw_states)
        states = np.clip((raw_states - self.state_rms.mean) / (self.state_rms.var**0.5 + 1e-8), -10, 10)
        next_states = np.clip((raw_next_states - self.state_rms.mean) / (self.state_rms.var**0.5 + 1e-8), -10, 10)
        
        states_t = torch.from_numpy(states).float().to(self.device)
        rewards_t = torch.from_numpy(raw_rewards).float().to(self.device)
        costs_t = torch.from_numpy(raw_costs).float().to(self.device)
        dones_t = torch.from_numpy(np.array(memory.dones[:memory.size])).float().to(self.device)
        next_states_t = torch.from_numpy(next_states).float().to(self.device)
        actions_t = torch.from_numpy(np.array(memory.actions[:memory.size])).float().to(self.device)

        values = self.actor_critic.get_value(states_t).squeeze()
        next_values = self.actor_critic.get_value(next_states_t).squeeze()
        cost_values = self.actor_critic.get_cost_value(states_t).squeeze()
        next_cost_values = self.actor_critic.get_cost_value(next_states_t).squeeze()

        N = len(rewards_t)
        returns, advantages = torch.zeros(N, device=self.device), torch.zeros(N, device=self.device)
        cost_returns, cost_advantages = torch.zeros(N, device=self.device), torch.zeros(N, device=self.device)
        gae, cost_gae = 0.0, 0.0

        for t in reversed(range(N)):
            mask = 1.0 - dones_t[t]
            
            # Correct reward delta calculation
            delta = rewards_t[t] + self.gamma * next_values[t] * mask - values[t]
            gae = delta + self.gamma * self.lam * mask * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
            
            # Correct cost delta calculation
            cost_delta = costs_t[t] + self.cost_gamma * next_cost_values[t] * mask - cost_values[t]
            cost_gae = cost_delta + self.cost_gamma * self.lam * mask * cost_gae 
            cost_advantages[t] = cost_gae
            cost_returns[t] = cost_gae + cost_values[t]

        log_probs_old = self.actor_critic.get_log_prob(states_t, actions_t)
        
        return {
            'states': states_t, 'actions': actions_t, 'log_probs_old': log_probs_old,
            'returns': returns, 'advantages': advantages,
            'cost_returns': cost_returns, 'cost_advantages': cost_advantages,
            'values_old': values, 'cost_values_old': cost_values,
            'raw_costs': raw_costs,
        }

    def update_parameters(self, memory, epochs, batch_size):
        data = self.process_data(memory)
        avg_rollout_cost = np.mean(data['raw_costs'])
        self.lagrange.update_lagrange_multiplier(avg_rollout_cost)
        # Add this before normalization in update_parameters
        if self.debug:
            print(f"Raw advantages - mean: {data['advantages'].mean():.6f}, std: {data['advantages'].std():.6f}")
            print(f"Raw advantages range: [{data['advantages'].min():.6f}, {data['advantages'].max():.6f}]")
        data['advantages'] = (data['advantages'] - data['advantages'].mean()) / (data['advantages'].std() + 1e-8)

        reward_policy_losses, cost_policy_losses = [], []
        value_losses, cost_value_losses = [], []
        clip_fractions, entropies = [], []
        
        # --- STAGE 1: REWARD MAXIMIZATION & CRITIC UPDATES ---
        stage1_dataset = TensorDataset(data['states'], data['actions'], data['log_probs_old'], 
                                       data['advantages'], data['returns'], data['cost_returns'],
                                       data['values_old'], data['cost_values_old'])
        stage1_loader = DataLoader(stage1_dataset, batch_size=self.batch_size, shuffle=True)

        if self.debug:
            with torch.no_grad():
                old_action_mean = self.actor_critic.get_distribution(data['states']).mean
        for epoch in range(epochs):
            for states_b, actions_b, log_probs_old_b, advantages_b, returns_b, cost_returns_b, values_old_b, cost_values_old_b in stage1_loader:
                log_probs, entropy, values, cost_values = self.actor_critic.evaluate(states_b, actions_b)
                
                
                # Actor update
                ratios = torch.exp(log_probs - log_probs_old_b)
                pi_loss_unclipped = ratios * advantages_b
                pi_loss_clipped = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_b
                reward_policy_loss = -torch.min(pi_loss_unclipped, pi_loss_clipped).mean() - self.entropy_coeff * entropy.mean()
                self.actor_optimizer.zero_grad()
                reward_policy_loss.backward()
                
                if self.debug:
                    total_grad_norm = 0
                    for p in self.actor_params:
                        if p.grad is not None:
                            total_grad_norm += p.grad.data.norm(2).item() ** 2
                    print(f"Actor gradient norm: {total_grad_norm ** 0.5}")
                                    
                    # print(f"Batch {batchidx}:")
                    print(f"  Advantages: mean={advantages_b.mean():.6f}, std={advantages_b.std():.6f}")
                    print(f"  Ratios: mean={ratios.mean():.6f}, std={ratios.std():.6f}, range=[{ratios.min():.6f}, {ratios.max():.6f}]")
                    print(f"  Policy loss unclipped: {pi_loss_unclipped.mean():.6f}")
                    print(f"  Policy loss clipped: {pi_loss_clipped.mean():.6f}")
                    print(f"  Final reward policy loss: {reward_policy_loss.item():.6f}")
                    print(f"  Log prob diff: {(log_probs - log_probs_old_b).abs().mean():.6f}")
                clip_grad_norm_(self.actor_params, self.max_grad_norm)
                self.actor_optimizer.step()

                # Critic update with value clipping
                values_clipped = values_old_b.reshape(-1, 1) + torch.clamp(values - values_old_b.reshape(-1, 1), -self.eps_clip, self.eps_clip)
                value_loss_clipped = F.mse_loss(values_clipped, returns_b.reshape(-1, 1))
                value_loss_unclipped = F.mse_loss(values, returns_b.reshape(-1, 1))
                value_loss = torch.max(value_loss_unclipped, value_loss_clipped)
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                clip_grad_norm_(self.critic_params, self.max_grad_norm)
                self.critic_optimizer.step()

                cost_values_clipped = cost_values_old_b.reshape(-1, 1) + torch.clamp(cost_values - cost_values_old_b.reshape(-1, 1), -self.eps_clip, self.eps_clip)
                cost_value_loss_clipped = F.mse_loss(cost_values_clipped, cost_returns_b.reshape(-1, 1))
                cost_value_loss_unclipped = F.mse_loss(cost_values, cost_returns_b.reshape(-1, 1))
                cost_value_loss = torch.max(cost_value_loss_unclipped, cost_value_loss_clipped)
                self.cost_critic_optimizer.zero_grad()
                cost_value_loss.backward()
                clip_grad_norm_(self.cost_critic_params, self.max_grad_norm)
                self.cost_critic_optimizer.step()

                with torch.no_grad():
                    reward_policy_losses.append(reward_policy_loss.item())
                    value_losses.append(value_loss.item())
                    cost_value_losses.append(cost_value_loss.item())
                    entropies.append(entropy.mean().item())
                    clip_fractions.append((torch.abs(ratios - 1.0) > self.eps_clip).float().mean().item())
                    

            with torch.no_grad():
                kl_div = (data['log_probs_old'] - self.actor_critic.get_log_prob(data['states'], data['actions'])).mean().item()
                if kl_div > self.trust_region_delta * 1.5:
                    break
        
        if self.debug:
            with torch.no_grad():
                new_action_mean = self.actor_critic.get_distribution(data['states']).mean
                param_change = torch.norm(new_action_mean - old_action_mean).item()
                print(f"Policy parameter change: {param_change}")
        # --- STAGE 2: COST CORRECTION ---
        with torch.no_grad():
            old_distribution = self.actor_critic.get_distribution(data['states'])

        cost_dataset = TensorDataset(data['states'], data['actions'], data['log_probs_old'], data['cost_advantages'], old_distribution.mean, old_distribution.stddev)
        cost_loader = DataLoader(cost_dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(epochs):
            for states_b, actions_b, logp_b, adv_c_b, old_mean_b, old_std_b in cost_loader:
                p_dist = Normal(old_mean_b, old_std_b)
                current_dist = self.actor_critic.get_distribution(states_b)
                logp_ = current_dist.log_prob(actions_b).sum(axis=-1)
                ratio = torch.exp(logp_ - logp_b)
                kl = torch.distributions.kl_divergence(current_dist, p_dist).mean()
                
                coef = (1 - self.gamma * self.lam) / (1 - self.gamma)
                cost_policy_loss = (self.lagrange.lagrangian_multiplier * coef * ratio * adv_c_b + kl).mean()
                
                self.actor_optimizer.zero_grad()
                cost_policy_loss.backward()
                clip_grad_norm_(self.actor_params, self.max_grad_norm)
                self.actor_optimizer.step()
                cost_policy_losses.append(cost_policy_loss.item())

            with torch.no_grad():
                kl_div_stage2 = torch.distributions.kl_divergence(self.actor_critic.get_distribution(data['states']), old_distribution).mean().item()
                if kl_div_stage2 > self.trust_region_delta * 1.5:
                    break

        with torch.no_grad():
            final_values = self.actor_critic.get_value(data['states']).squeeze()
            final_cost_values = self.actor_critic.get_cost_value(data['states']).squeeze()
            var_y = torch.var(data['returns'])
            explained_var_value = (1 - torch.var(data['returns'] - final_values) / (var_y + 1e-8)).item()
            var_y_cost = torch.var(data['cost_returns'])
            explained_var_cost_value = (1 - torch.var(data['cost_returns'] - final_cost_values) / (var_y_cost + 1e-8)).item()

        memory.clear_memory()

        return {
            "avg_reward_policy_loss": np.mean(reward_policy_losses),
            "avg_cost_policy_loss": np.mean(cost_policy_losses),
            "avg_value_loss": np.mean(value_losses),
            "avg_cost_value_loss": np.mean(cost_value_losses),
            "multiplier": self.lagrange.lagrangian_multiplier.item(),
            "clip_fraction": np.mean(clip_fractions),
            "entropy": np.mean(entropies),
            "explained_var_value": explained_var_value,
            "explained_var_cost_value": explained_var_cost_value,
            "kl_divergence": (kl_div_stage2 + kl_div) / 2,
            "avg_rollout_cost": avg_rollout_cost,
        }

    def save_checkpoint(self, path):
        torch.save(self.actor_critic.state_dict(), path)

    def load_checkpoint(self, path):
        self.actor_critic.load_state_dict(torch.load(path, map_location=self.device))
