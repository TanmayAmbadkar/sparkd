import torch
import torch.nn.functional as F
from torch.optim import Adam
from cpo.model import CPOActorCritic
import numpy as np
from sklearn.metrics import explained_variance_score

class CUP:
    def __init__(self, obs_dim, action_space, args):
        self.gamma = getattr(args, "gamma", 0.99)
        self.cost_gamma = getattr(args, "cost_gamma", 0.99)
        self.lam = getattr(args, "lam", 0.95)
        self.eps_clip = getattr(args, "eps_clip", 0.2)
        self.value_coeff = getattr(args, "value_coeff", 0.5)
        self.entropy_coeff = getattr(args, "entropy_coeff", 0.01)
        self.device = torch.device("cuda" if getattr(args, "cuda", False) else "cpu")

        # CUP-specific
        self.trust_region_delta = getattr(args, "cup_trust_region", 0.01)   # Trust region (KL)
        self.use_bias_corrected_gae = getattr(args, "cup_bias_corrected_gae", False)
        self.use_lagrangian = getattr(args, "cup_lagrangian", True)         # If True, uses primal-dual for multiplier

        # Single constraint only
        self.cost_limit = float(getattr(args, "cost_limit", 10.0))
        self.multiplier = float(getattr(args, "lagrange_init", 1.0))
        self.lagrange_lr = getattr(args, "lagrange_lr", 0.01)

        # Network and optimizers
        self.actor_critic = CPOActorCritic(obs_dim, action_space, args.hidden_size).to(self.device)
        self.actor_params = list(self.actor_critic.actor.parameters()) + [self.actor_critic.actor_logstd]
        self.critic_params = list(self.actor_critic.critic.parameters())
        self.cost_critic_params = list(self.actor_critic.cost_critic.parameters())
        self.actor_optimizer = torch.optim.Adam(self.actor_params, lr=args.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic_params, lr=args.lr)
        self.cost_critic_optimizer = torch.optim.Adam(self.cost_critic_params, lr=args.lr)

    def select_action(self, state):
        state = torch.Tensor(state).to(self.device).unsqueeze(0)
        action, log_prob = self.actor_critic.act(state)
        return action.cpu().detach().numpy()[0], log_prob.cpu().detach().numpy()[0]

    @torch.no_grad()
    def log_prob_advantage_estimation(self, memory):
        states = memory.states[:memory.size]
        rewards = memory.rewards[:memory.size]
        costs = np.array(memory.costs[:memory.size])   # shape (N,)
        dones = memory.dones[:memory.size]
        next_states = memory.next_states[:memory.size]
        actions = memory.actions[:memory.size]

        state_tensor = torch.Tensor(states).to(self.device)
        next_state_tensor = torch.Tensor(next_states).to(self.device)
        actions = torch.Tensor(actions).to(self.device)
        with torch.no_grad():
            values = self.actor_critic.get_value(state_tensor).squeeze().detach()
            next_values = self.actor_critic.get_value(next_state_tensor).squeeze().detach()
            cost_values = self.actor_critic.get_cost_value(state_tensor).squeeze().detach()
            next_cost_values = self.actor_critic.get_cost_value(next_state_tensor).squeeze().detach()

        N = len(rewards)
        returns = torch.zeros(N)
        advantages = torch.zeros(N)
        cost_returns = torch.zeros(N)
        cost_advantages = torch.zeros(N)
        gae = 0.0
        cost_gae = 0.0

        for t in reversed(range(N)):
            mask = 1 - dones[t]
            delta = rewards[t] + self.gamma * next_values[t] * mask - values[t]
            gae = delta + self.gamma * self.lam * mask * gae
            advantages[t] = gae
            returns[t] = gae + values[t]

            cost_delta = costs[t] + self.cost_gamma * next_cost_values[t] * mask - cost_values[t]
            cost_gae = cost_delta + self.cost_gamma * self.lam * mask * cost_gae
            cost_advantages[t] = cost_gae
            cost_returns[t] = cost_gae + cost_values[t]

        # Bias-corrected GAE (optional)
        if self.use_bias_corrected_gae:
            advantages -= advantages.mean()
            cost_advantages -= cost_advantages.mean()

        return (state_tensor, actions, self.actor_critic.get_log_prob(state_tensor, actions),
                returns, advantages, cost_returns, cost_advantages, self.actor_critic.get_value(state_tensor).squeeze().detach(),
                self.actor_critic.get_cost_value(state_tensor).squeeze().detach())

    def cup_surrogate(self, ratios, advantages, kl, delta):
        c = (advantages.abs() / delta).detach()
        surrogate = ratios * advantages - c * kl
        return surrogate.mean()
    
    def update_parameters(self, memory, epochs, batch_size):
        # --- INITIAL SETUP ---
        (states, actions, log_probs_old, returns, advantages,
         cost_returns, cost_advantages, values_old, cost_values_old) = self.log_prob_advantage_estimation(memory)

        # Normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # We don't normalize cost advantages in this approach, as their magnitude is meaningful for the multiplier
        
        # --- LAGRANGE MULTIPLIER UPDATE (Once per rollout) ---
        if self.use_lagrangian:
            # Calculate average cost from the completed rollout
            avg_cost = torch.mean(cost_values_old) 
            with torch.no_grad():
                self.multiplier = max(
                    0.0,
                    # Note: You might need a separate, smaller learning rate for this style of update
                    self.multiplier + self.lagrange_lr * (avg_cost - self.cost_limit) 
                )

        # Logging variables
        num_updates = 0
        total_reward_policy_loss = 0.0
        total_cost_policy_loss = 0.0
        total_value_loss = 0.0
        total_cost_value_loss = 0.0

        # ========================================================================================
        # --- STAGE 1: REWARD MAXIMIZATION (PPO-Clip Update) & CRITIC UPDATES ---
        # ========================================================================================
        for epoch in range(epochs):
            for idx in range(0, len(states), batch_size):
                batch_slice = slice(idx, idx + batch_size)

                # Policy evaluation
                log_probs, entropy, values, cost_values = self.actor_critic.evaluate(
                    states[batch_slice], actions[batch_slice]
                )

                # --- Policy Update (PPO-Clip for Reward) ---
                ratios = torch.exp(log_probs - log_probs_old[batch_slice])
                
                # Unclipped reward objective
                pi_loss_unclipped = ratios * advantages[batch_slice]
                
                # Clipped reward objective
                pi_loss_clipped = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages[batch_slice]
                
                # Final PPO policy loss for rewards
                # We add the entropy bonus to encourage exploration
                reward_policy_loss = -torch.min(pi_loss_unclipped, pi_loss_clipped).mean() - self.entropy_coeff * entropy.mean()
                
                self.actor_optimizer.zero_grad()
                reward_policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_params, 0.5)
                self.actor_optimizer.step()

                # --- Critic Updates (with Clipped Value Loss) ---
                # Reward Critic
                self.critic_optimizer.zero_grad()
                loss_v_unclipped = (values - returns[batch_slice].reshape(-1, 1)) ** 2
                values_clipped = values_old[batch_slice].reshape(-1, 1) + torch.clamp(
                    values - values_old[batch_slice].reshape(-1, 1), -0.2, 0.2
                )
                loss_v_clipped = (values_clipped - returns[batch_slice].reshape(-1, 1)) ** 2
                value_loss = 0.5 * torch.mean(torch.max(loss_v_unclipped, loss_v_clipped))
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic_params, 0.5)
                self.critic_optimizer.step()

                # Cost Critic
                self.cost_critic_optimizer.zero_grad()
                loss_c_unclipped = (cost_values - cost_returns[batch_slice].reshape(-1, 1)) ** 2
                cost_values_clipped = cost_values_old[batch_slice].reshape(-1, 1) + torch.clamp(
                    cost_values - cost_values_old[batch_slice].reshape(-1, 1), -0.2, 0.2
                )
                loss_c_clipped = (cost_values_clipped - cost_returns[batch_slice].reshape(-1, 1)) ** 2
                cost_value_loss = 0.5 * torch.mean(torch.max(loss_c_unclipped, loss_c_clipped))
                cost_value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.cost_critic_params, 0.5)
                self.cost_critic_optimizer.step()

                # Logging
                total_reward_policy_loss += reward_policy_loss.item()
                total_value_loss += value_loss.item()
                total_cost_value_loss += cost_value_loss.item()
                num_updates += 1

            # KL-based early stopping for Stage 1
            with torch.no_grad():
                log_probs_new, _, _, _ = self.actor_critic.evaluate(states, actions)
                kl_div = torch.mean(log_probs_old - log_probs_new).item()
                if kl_div > self.trust_region_delta * 1.5:
                    # print(f"Stopping Stage 1 early at epoch {epoch+1}, KL: {kl_div:.4f}")
                    break
        
        # ========================================================================================
        # --- STAGE 2: COST CORRECTION (KL-Penalized Cost Minimization) ---
        # ========================================================================================
        
        # Get the log_probs from the policy at the end of Stage 1
        with torch.no_grad():
            log_probs_stage1_end, _, _, _ = self.actor_critic.evaluate(states, actions)
            
        for epoch in range(epochs): # A separate set of epochs for the cost update
            for idx in range(0, len(states), batch_size):
                batch_slice = slice(idx, idx + batch_size)

                # Policy evaluation
                log_probs, _, _, _ = self.actor_critic.evaluate(
                    states[batch_slice], actions[batch_slice]
                )
                
                # --- Policy Update (KL-Penalized Cost Correction) ---
                ratios = torch.exp(log_probs - log_probs_stage1_end[batch_slice])

                # The KL divergence is between the current policy and the policy from the end of Stage 1
                kl_new_old_stage2 = torch.mean(log_probs_stage1_end[batch_slice] - log_probs)
                
                # Cost advantage objective
                cost_objective = (ratios * cost_advantages[batch_slice]).mean()
                
                # Combined loss for cost correction
                cost_policy_loss = self.multiplier * cost_objective + kl_new_old_stage2
                
                self.actor_optimizer.zero_grad()
                cost_policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_params, 0.5)
                self.actor_optimizer.step()

                total_cost_policy_loss += cost_policy_loss.item()
            
            # KL-based early stopping for Stage 2
            with torch.no_grad():
                log_probs_new, _, _, _ = self.actor_critic.evaluate(states, actions)
                kl_div = torch.mean(log_probs_stage1_end - log_probs_new).item()
                if kl_div > self.trust_region_delta * 1.5:
                    # print(f"Stopping Stage 2 early at epoch {epoch+1}, KL: {kl_div:.4f}")
                    break

        memory.clear_memory()

        # Update and return average losses for logging
        avg_reward_policy_loss = total_reward_policy_loss / num_updates if num_updates > 0 else 0
        avg_cost_policy_loss = total_cost_policy_loss / num_updates if num_updates > 0 else 0
        avg_value_loss = total_value_loss / num_updates if num_updates > 0 else 0
        avg_cost_value_loss = total_cost_value_loss / num_updates if num_updates > 0 else 0

        return {
            "avg_reward_policy_loss": avg_reward_policy_loss,
            "avg_cost_policy_loss": avg_cost_policy_loss,
            "avg_value_loss": avg_value_loss,
            "avg_cost_value_loss": avg_cost_value_loss,
            "multiplier": self.multiplier,
        }

        memory.clear_memory()

        avg_policy_loss = total_policy_loss / num_updates
        avg_value_loss = total_value_loss / num_updates
        avg_cost_value_loss = total_cost_value_loss / num_updates
        avg_entropy_loss = total_entropy_loss / num_updates
        avg_total_loss = total_loss / num_updates
        avg_clip_fraction = total_clip_frac / num_updates
        avg_kl_divergence = total_kl_div / num_updates
        avg_explained_var = total_explained_var / num_updates

        return {
            "avg_policy_loss": avg_policy_loss,
            "avg_value_loss": avg_value_loss,
            "avg_cost_value_loss": avg_cost_value_loss,
            "avg_entropy_loss": avg_entropy_loss,
            "avg_total_loss": avg_total_loss,
            "avg_clip_fraction": avg_clip_fraction,
            "avg_kl_divergence": avg_kl_divergence,
            "avg_explained_variance": avg_explained_var,
            "trust_region_delta": self.trust_region_delta,
            "multiplier": self.multiplier,
        }

    def save_checkpoint(self, path):
        torch.save(self.actor_critic.state_dict(), path)

    def load_checkpoint(self, path):
        self.actor_critic.load_state_dict(torch.load(path, map_location=self.device))
