import torch
import torch.nn.functional as F
from torch.optim import Adam
from cpo.model import CPOActorCritic # Assuming this is your model definition
import numpy as np
from cpo.utils import RunningMeanStd
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_

class PCRPO:
    def __init__(self, obs_dim, action_space, args):
        self.gamma = getattr(args, "gamma", 0.99)
        self.cost_gamma = getattr(args, "cost_gamma", 0.99)
        self.lam = getattr(args, "lam", 0.95)
        self.entropy_coeff = getattr(args, "entropy_coeff", 0.01)
        self.device = torch.device("cuda" if getattr(args, "cuda", True) else "cpu")
        self.max_grad_norm = getattr(args, "max_grad_norm", 0.5)
        self.debug = getattr(args, "debug", False)
        print(f"Using device: {self.device}")

        # --- PCRPO-specific parameters from paper ---
        self.batch_size = getattr(args, "mini_batch_size", 128)
        self.cost_limit = float(getattr(args, "cost_limit", 10.0))
        
        # Slack bounds from Algorithm 1
        self.h_plus = getattr(args, "h_plus", 5.0)
        self.h_minus = getattr(args, "h_minus", -5.0)

        # Actor-critic networks and optimizers
        self.actor_critic = CPOActorCritic(obs_dim, action_space, args.hidden_size).to(self.device)
        self.actor_params = list(self.actor_critic.actor.parameters()) + [self.actor_critic.actor_logstd]
        self.critic_params = list(self.actor_critic.critic.parameters())
        self.cost_critic_params = list(self.actor_critic.cost_critic.parameters())
        self.actor_optimizer = Adam(self.actor_params, lr=args.actor_lr)
        self.critic_optimizer = Adam(self.critic_params, lr=args.critic_lr)
        self.cost_critic_optimizer = Adam(self.cost_critic_params, lr=args.critic_lr)
        
        # State normalization
        self.state_rms = RunningMeanStd(shape=obs_dim)

    @torch.no_grad()
    def select_action(self, state):
        """Selects an action given a state, using the running mean-std for normalization."""
        state_normalized = np.clip((state - self.state_rms.mean) / (self.state_rms.var**0.5 + 1e-8), -10, 10)
        state_tensor = torch.from_numpy(state_normalized).float().to(self.device).unsqueeze(0)
        action, log_prob = self.actor_critic.act(state_tensor)
        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0]

    @torch.no_grad()
    def process_data(self, memory):
        """Processes rollout data to compute advantages and returns for both reward and cost."""
        raw_states = np.array(memory.states[:memory.size])
        raw_rewards = np.array(memory.rewards[:memory.size])
        raw_costs = np.array(memory.costs[:memory.size])
        raw_next_states = np.array(memory.next_states[:memory.size])

        # Update and apply state normalization
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

        # Compute GAE
        N = len(rewards_t)
        returns, advantages = torch.zeros(N, device=self.device), torch.zeros(N, device=self.device)
        cost_returns, cost_advantages = torch.zeros(N, device=self.device), torch.zeros(N, device=self.device)
        gae, cost_gae = 0.0, 0.0

        for t in reversed(range(N)):
            mask = 1.0 - dones_t[t]
            
            delta = rewards_t[t] + self.gamma * next_values[t] * mask - values[t]
            gae = delta + self.gamma * self.lam * mask * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
            
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

    def _get_gradients(self, ratios, advantages_b, cost_advantages_b):
        """Helper to compute reward and cost gradient vectors for a given batch."""
        reward_surrogate = (ratios * advantages_b).mean()
        cost_surrogate = (ratios * cost_advantages_b).mean()

        # Get reward gradient (for ascent)
        self.actor_optimizer.zero_grad()
        (-reward_surrogate).backward(retain_graph=True)
        g_r = torch.cat([p.grad.flatten() for p in self.actor_params if p.grad is not None])

        # Get cost gradient (for descent)
        self.actor_optimizer.zero_grad()
        cost_surrogate.backward(retain_graph=True)
        g_c_raw = torch.cat([p.grad.flatten() for p in self.actor_params if p.grad is not None])
        g_c_descent = -g_c_raw

        return g_r, g_c_descent, reward_surrogate, cost_surrogate

    def _apply_gradient(self, final_grad):
        """Helper to manually set parameter gradients and step the optimizer."""
        offset = 0
        for p in self.actor_params:
            if p.grad is not None:
                p_shape = p.grad.shape
                p_size = p.grad.numel()
                p.grad.copy_(final_grad[offset : offset + p_size].view(p_shape))
                offset += p_size
        
        clip_grad_norm_(self.actor_params, self.max_grad_norm)
        self.actor_optimizer.step()


    def update_parameters(self, memory, epochs):
        """Main update function with slack-based framework and detailed logging."""
        data = self.process_data(memory)
        
        # Normalize advantages
        data['advantages'] = (data['advantages'] - data['advantages'].mean()) / (data['advantages'].std() + 1e-8)
        data['cost_advantages'] = (data['cost_advantages'] - data['cost_advantages'].mean()) / (data['cost_advantages'].std() + 1e-8)

        # Determine update strategy from Algorithm 1
        avg_rollout_cost = np.mean(data['raw_costs'])
        update_type = ""
        if avg_rollout_cost > self.cost_limit + self.h_plus:
            update_type = "cost_only"
        elif avg_rollout_cost < self.cost_limit + self.h_minus:
            update_type = "reward_only"
        else:
            update_type = "combined"

        # --- Initialize tracking lists ---
        policy_losses, value_losses, cost_value_losses = [], [], []
        reward_surrogates, cost_surrogates, cosine_similarities, entropies = [], [], [], []
        
        dataset = TensorDataset(data['states'], data['actions'], data['log_probs_old'], 
                                data['advantages'], data['returns'], data['cost_returns'],
                                data['cost_advantages'])
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(epochs):
            for states_b, actions_b, log_probs_old_b, advantages_b, returns_b, cost_returns_b, cost_advantages_b in dataloader:
                
                log_probs, entropy, values, cost_values = self.actor_critic.evaluate(states_b, actions_b)
                ratios = torch.exp(log_probs - log_probs_old_b)
                
                # --- Actor Update Logic ---
                final_update_direction = None
                policy_loss = torch.tensor(0.)

                if update_type == "combined":
                    g_r, g_c_descent, rs, cs = self._get_gradients(ratios, advantages_b, cost_advantages_b)
                    cos_sim = F.cosine_similarity(g_r, g_c_descent, dim=0, eps=1e-8)
                    
                    if cos_sim < 0: # Conflict case
                        g_r_proj = (torch.dot(g_r, g_c_descent) / torch.dot(g_c_descent, g_c_descent)) * g_c_descent
                        g_r_plus = g_r - g_r_proj
                        g_c_descent_proj = (torch.dot(g_c_descent, g_r) / torch.dot(g_r, g_r)) * g_r
                        g_c_descent_plus = g_c_descent - g_c_descent_proj
                        final_update_direction = 0.5 * g_r_plus + 0.5 * g_c_descent_plus
                    else: # Non-conflict case
                        final_update_direction = 0.5 * g_r + 0.5 * g_c_descent

                    final_grad = -final_update_direction
                    self._apply_gradient(final_grad)
                    
                    reward_surrogates.append(rs.item())
                    cost_surrogates.append(cs.item())
                    cosine_similarities.append(cos_sim.item())
                else:
                    reward_surrogate = (ratios * advantages_b).mean()
                    cost_surrogate = (ratios * cost_advantages_b).mean()

                    if update_type == "reward_only":
                        policy_loss = -reward_surrogate - self.entropy_coeff * entropy.mean()
                    elif update_type == "cost_only":
                        policy_loss = cost_surrogate - self.entropy_coeff * entropy.mean()
                    
                    self.actor_optimizer.zero_grad()
                    policy_loss.backward()
                    clip_grad_norm_(self.actor_params, self.max_grad_norm)
                    self.actor_optimizer.step()
                    
                    reward_surrogates.append(reward_surrogate.item())
                    cost_surrogates.append(cost_surrogate.item())

                # --- Debugging Printouts ---
                if self.debug:
                    print(f"\n--- Batch Debug Info (Update Type: {update_type}) ---")
                    if final_update_direction is not None: # Combined case
                        print(f"  Cosine Similarity: {cos_sim.item():.6f}")
                        print(f"  Gradient Norms: Reward={torch.linalg.norm(g_r):.4f}, Cost-Descent={torch.linalg.norm(g_c_descent):.4f}, Final={torch.linalg.norm(final_update_direction):.4f}")
                    else: # Reward/Cost only cases
                        grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.actor_params if p.grad is not None) ** 0.5
                        print(f"  Policy Loss: {policy_loss.item():.6f}")
                        print(f"  Actor Grad Norm: {grad_norm:.4f}")
                    print(f"  Surrogates: Reward={reward_surrogates[-1]:.6f}, Cost={cost_surrogates[-1]:.6f}")

                # --- Critic Updates (identical for all cases) ---
                value_loss = F.mse_loss(values.squeeze(), returns_b)
                cost_value_loss = F.mse_loss(cost_values.squeeze(), cost_returns_b)
                
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                clip_grad_norm_(self.critic_params, self.max_grad_norm)
                self.critic_optimizer.step()
                
                self.cost_critic_optimizer.zero_grad()
                cost_value_loss.backward()
                clip_grad_norm_(self.cost_critic_params, self.max_grad_norm)
                self.cost_critic_optimizer.step()

                # --- Append to tracking lists ---
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                cost_value_losses.append(cost_value_loss.item())
                entropies.append(entropy.mean().item())
        
        # --- Final Logging and Cleanup ---
        with torch.no_grad():
            final_values = self.actor_critic.get_value(data['states']).squeeze()
            final_cost_values = self.actor_critic.get_cost_value(data['states']).squeeze()
            var_y = torch.var(data['returns'])
            explained_var_value = (1 - torch.var(data['returns'] - final_values) / (var_y + 1e-8)).item()
            var_y_cost = torch.var(data['cost_returns'])
            explained_var_cost_value = (1 - torch.var(data['cost_returns'] - final_cost_values) / (var_y_cost + 1e-8)).item()

        memory.clear_memory()

        return {
            "avg_policy_loss": np.mean(policy_losses),
            "avg_reward_surrogate": np.mean(reward_surrogates),
            "avg_cost_surrogate": np.mean(cost_surrogates),
            "avg_cosine_similarity": np.mean(cosine_similarities) if cosine_similarities else None,
            "avg_value_loss": np.mean(value_losses),
            "avg_cost_value_loss": np.mean(cost_value_losses),
            "entropy": np.mean(entropies),
            "explained_var_value": explained_var_value,
            "explained_var_cost_value": explained_var_cost_value,
            "avg_rollout_cost": avg_rollout_cost,
            "update_type": update_type,
        }

    def save_checkpoint(self, path):
        torch.save(self.actor_critic.state_dict(), path)

    def load_checkpoint(self, path):
        self.actor_critic.load_state_dict(torch.load(path, map_location=self.device))