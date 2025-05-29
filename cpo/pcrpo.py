import torch
import torch.nn.functional as F
from torch.optim import Adam
from cpo.model import CPOActorCritic   # Import your own actor-critic with cost_critic
import numpy as np
from sklearn.metrics import explained_variance_score

class PCRPO:
    def __init__(self, obs_dim, action_space, args):
        self.gamma = getattr(args, "gamma", 0.99)
        self.cost_gamma = getattr(args, "cost_gamma", 0.99)
        self.lam = getattr(args, "lam", 0.95)
        self.eps_clip = getattr(args, "eps_clip", 0.2)
        self.value_coeff = getattr(args, "value_coeff", 0.5)
        self.entropy_coeff = getattr(args, "entropy_coeff", 0.01)
        self.device = torch.device("cuda" if getattr(args, "cuda", False) else "cpu")

        self.switching_temp = getattr(args, "switching_temp", 1.0)  # PCRPO hyperparam
        self.cost_limit = getattr(args, "cost_limit", 10.0)
        # self.slack_coef = getattr(args, "slack_coef", 1.0)  # Optional

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

    def compute_switching_coeff(self, grad_reward, grad_cost, temp=1.0):
        # Cosine similarity between reward and cost gradients
        cos_sim = F.cosine_similarity(grad_reward, grad_cost, dim=0, eps=1e-8)
        # Soft switching coefficient (PCRPO)
        coeff = torch.sigmoid(-cos_sim / temp)
        return coeff

    def _get_grad_vector(self, params):
        # Utility: return parameter gradients as a single flat vector
        grads = []
        for p in params:
            if p.grad is not None:
                grads.append(p.grad.view(-1))
        if grads:
            return torch.cat(grads)
        else:
            # In case none of the params had gradients
            return torch.zeros(1, device=self.device)

    def update_parameters(self, memory, epochs, batch_size):
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

                # Policy evaluation
                log_probs, entropy, values, cost_values = self.actor_critic.evaluate(
                    states[batch_slice], actions[batch_slice]
                )
                ratios = torch.exp(log_probs - log_probs_old[batch_slice])

                norm_advantages = (advantages[batch_slice] - advantages[batch_slice].mean()) / (advantages[batch_slice].std() + 1e-8)
                norm_cost_advantages = (cost_advantages[batch_slice] - cost_advantages[batch_slice].mean()) / (cost_advantages[batch_slice].std() + 1e-8)

                reward_surrogate = -torch.mean(ratios * norm_advantages.reshape(-1, 1))
                cost_surrogate = torch.mean(ratios * norm_cost_advantages.reshape(-1, 1))

                # --- Compute reward/cost gradients w.r.t actor parameters only ---
                grad_reward = torch.autograd.grad(
                    reward_surrogate, self.actor_params, retain_graph=True, create_graph=False, allow_unused=True
                )
                grad_cost = torch.autograd.grad(
                    cost_surrogate, self.actor_params, retain_graph=True, create_graph=False, allow_unused=True
                )
                grad_reward_vec = torch.cat([
                    g.reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1)
                    for g, p in zip(grad_reward, self.actor_params)
                ])
                grad_cost_vec = torch.cat([
                    g.reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1)
                    for g, p in zip(grad_cost, self.actor_params)
                ])

                coeff = self.compute_switching_coeff(grad_reward_vec, grad_cost_vec, temp=self.switching_temp)

                # --- Do synthetic loss update for actor only ---
                self.actor_optimizer.zero_grad()
                synthetic_loss = reward_surrogate * (1 - coeff) + cost_surrogate * coeff
                synthetic_loss.backward()
                self.actor_optimizer.step()

                # --- Value Critic update ---
                self.critic_optimizer.zero_grad()
                value_loss = F.mse_loss(values, returns[batch_slice].reshape(-1, 1))
                value_loss.backward()
                self.critic_optimizer.step()

                # --- Cost Critic update ---
                self.cost_critic_optimizer.zero_grad()
                cost_value_loss = F.mse_loss(cost_values, cost_returns[batch_slice].reshape(-1, 1))
                cost_value_loss.backward()
                self.cost_critic_optimizer.step()

                # Logging
                entropy_loss = entropy.mean()
                clipped_mask = (ratios > (1 + self.eps_clip)) | (ratios < (1 - self.eps_clip))
                clip_fraction = torch.mean(clipped_mask.float()).item()
                total_clip_frac += clip_fraction

                kl_div = torch.mean(log_probs_old[batch_slice] - log_probs).item()
                total_kl_div += kl_div

                explained_variance = explained_variance_score(
                    returns[batch_slice].detach().cpu().numpy(),
                    values.detach().cpu().numpy(),
                )
                total_explained_var += explained_variance

                total_policy_loss += synthetic_loss.item()
                total_value_loss += value_loss.item()
                total_cost_value_loss += cost_value_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_loss += (synthetic_loss + value_loss + cost_value_loss).item()
                num_updates += 1

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
            "switching_temp": self.switching_temp,
        }


    def save_checkpoint(self, path):
        torch.save(self.actor_critic.state_dict(), path)

    def load_checkpoint(self, path):
        self.actor_critic.load_state_dict(torch.load(path, map_location=self.device))
