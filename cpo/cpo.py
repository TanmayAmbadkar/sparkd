import torch
import torch.nn.functional as F
from torch.optim import Adam
from cpo.model import CPOActorCritic   # Import the CPO-compatible model with cost_critic
import numpy as np
from sklearn.metrics import explained_variance_score

class CPO:
    def __init__(self, obs_dim, action_space, args):
        self.gamma = 0.99
        self.cost_gamma = 0.99  # Discount for cost
        self.lam = 0.95  # GAE lambda
        self.eps_clip = 0.2
        self.value_coeff = 0.5
        self.entropy_coeff = 0.01
        self.cost_limit = args.cost_limit
        self.lagrange_multiplier = args.lagrange_init if hasattr(args, "lagrange_init") else 1.0
        self.lagrange_lr = getattr(args, "lagrange_lr", 0.01)
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.actor_critic = CPOActorCritic(obs_dim, action_space, args.hidden_size).to(self.device)
        self.optimizer = Adam(self.actor_critic.parameters(), lr=args.lr)
        self.action_space = action_space

    def select_action(self, state):
        state = torch.Tensor(state).to(self.device).unsqueeze(0)
        action, log_prob = self.actor_critic.act(state)
        return action.cpu().detach().numpy()[0], log_prob.cpu().detach().numpy()[0]

    @torch.no_grad()
    def log_prob_advantage_estimation(self, memory):
        """
        Compute returns and advantages for reward and cost using GAE.
        Assumes memory has attributes:
          - states, rewards, costs, dones, next_states, actions
        """
        states = memory.states[:memory.size]
        rewards = memory.rewards[:memory.size]
        costs = memory.costs[:memory.size]
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

        return (state_tensor, actions, self.actor_critic.get_log_prob(state_tensor, actions),
                returns, advantages, cost_returns, cost_advantages)

    def update_parameters(self, memory, epochs, batch_size):
        # Compute returns and advantages for both reward and cost
        (states, actions, log_probs_old, returns, advantages, 
         cost_returns, cost_advantages) = self.log_prob_advantage_estimation(memory)

        # Normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        cost_advantages = (cost_advantages - cost_advantages.mean()) / (cost_advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_cost_value_loss = 0.0
        total_entropy_loss = 0.0
        total_loss = 0.0
        total_clip_frac = 0.0
        total_kl_div = 0.0
        total_explained_var = 0.0
        num_updates = 0

        for _ in range(epochs):
            for idx in range(0, len(states), batch_size):
                batch_slice = slice(idx, idx + batch_size)

                # Evaluate current policy and value function
                log_probs, entropy, values, cost_values = self.actor_critic.evaluate(
                    states[batch_slice], actions[batch_slice]
                )

                # Probability ratios
                ratios = torch.exp(log_probs - log_probs_old[batch_slice])
                clipped_mask = (ratios > (1 + self.eps_clip)) | (ratios < (1 - self.eps_clip))
                clip_fraction = torch.mean(clipped_mask.float()).item()
                total_clip_frac += clip_fraction

                # Approximate KL divergence as the mean difference between old and new log probabilities
                kl_div = torch.mean(log_probs_old[batch_slice] - log_probs).item()
                total_kl_div += kl_div

                # Normalize batch
                norm_advantages = (advantages[batch_slice] - advantages[batch_slice].mean()) / (advantages[batch_slice].std() + 1e-8)
                norm_cost_advantages = (cost_advantages[batch_slice] - cost_advantages[batch_slice].mean()) / (cost_advantages[batch_slice].std() + 1e-8)

                # --- CPO/Lagrangian penalty surrogate ---
                # surr1 = ratios * norm_advantages.reshape(-1, 1)
                # surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * norm_advantages.reshape(-1, 1)
                # policy_loss = -torch.min(surr1, surr2).mean()

                # For full CPO, replace this block with the constrained update/projection per Achiam et al. (2017)
                # For practical code, use Lagrangian penalty (as in most CPO codebases)
                cost_surrogate = (ratios * norm_cost_advantages.reshape(-1, 1)).mean()

                # Lagrangian surrogate
                surr1 = ratios * norm_advantages.reshape(-1, 1)
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * norm_advantages.reshape(-1, 1)
                policy_loss = -torch.min(surr1, surr2).mean() + self.lagrange_multiplier * cost_surrogate

                # Value loss for reward and cost critics
                value_loss = F.mse_loss(values, returns[batch_slice].reshape(-1, 1))
                cost_value_loss = F.mse_loss(cost_values, cost_returns[batch_slice].reshape(-1, 1))
                entropy_loss = entropy.mean()

                # Full loss
                loss = (policy_loss +
                        self.value_coeff * value_loss +
                        self.value_coeff * cost_value_loss -
                        self.entropy_coeff * entropy_loss)

                explained_variance = explained_variance_score(
                    returns[batch_slice].detach().cpu().numpy(),
                    values.detach().cpu().numpy(),
                )
                total_explained_var += explained_variance.item()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_cost_value_loss += cost_value_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_loss += loss.item()
                num_updates += 1

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # ---- Update Lagrange Multiplier (for penalty version) ----
                batch_mean_cost = cost_surrogate.item()
                # Soft update for the multiplier (projected to >= 0)
                self.lagrange_multiplier = max(
                    0.0,
                    self.lagrange_multiplier + self.lagrange_lr * (batch_mean_cost - self.cost_limit)
                )

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
            "lagrange_multiplier": self.lagrange_multiplier,
        }

    def save_checkpoint(self, path):
        torch.save(self.actor_critic.state_dict(), path)

    def load_checkpoint(self, path):
        self.actor_critic.load_state_dict(torch.load(path, map_location=self.device))
