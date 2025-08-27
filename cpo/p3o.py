import torch
import torch.nn.functional as F
from torch.optim import Adam
from cpo.model import CPOActorCritic
import numpy as np
from cpo.utils import RunningMeanStd
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_

class P3O:
    def __init__(self, obs_dim, action_space, args):
        self.gamma = getattr(args, "gamma", 0.99)
        self.cost_gamma = getattr(args, "cost_gamma", 0.99)
        self.lam = getattr(args, "lam", 0.95)
        self.eps_clip = getattr(args, "eps_clip", 0.2)
        self.entropy_coeff = getattr(args, "entropy_coeff", 0.01)
        self.device = torch.device("cuda" if getattr(args, "cuda", True) else "cpu")
        self.max_grad_norm = getattr(args, "max_grad_norm", 0.5)
        print(f"Using device: {self.device}")

        # P3O-specific parameters
        self.kappa = getattr(args, "p3o_kappa", 1.0)  # Fixed penalty coefficient
        self.batch_size = getattr(args, "mini_batch_size", 64)
        self.cost_limit = float(getattr(args, "cost_limit", 0.0))

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

        # Calculate mean episodic cost for P3O
        ep_costs = []
        current_ep_cost = 0
        for i in range(len(raw_costs)):
            current_ep_cost += raw_costs[i]
            if memory.dones[i]:
                ep_costs.append(current_ep_cost)
                current_ep_cost = 0
        if not memory.dones[-1]:
            ep_costs.append(current_ep_cost)
        mean_ep_cost = np.mean(ep_costs) if ep_costs else 0

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
            'mean_ep_cost': mean_ep_cost,
            'raw_costs': raw_costs,
        }

    def _loss_pi_reward(self, log_probs, log_probs_old, advantages):
        """Standard PPO clipped surrogate objective for rewards."""
        ratios = torch.exp(log_probs - log_probs_old)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        return -torch.min(surr1, surr2).mean()

    def _loss_pi_cost(self, log_probs, log_probs_old, cost_advantages, mean_ep_cost):
        """P3O-specific cost penalty loss."""
        ratios = torch.exp(log_probs - log_probs_old)
        surr_cadv = (ratios * cost_advantages).mean()
        
        jc = mean_ep_cost - self.cost_limit
        loss_cost = self.kappa * F.relu(surr_cadv + jc)
        return loss_cost

    def update_parameters(self, memory, epochs, batch_size):
        data = self.process_data(memory)
        
        if self.debug:
            print(f"Raw advantages - mean: {data['advantages'].mean():.6f}, std: {data['advantages'].std():.6f}")
            print(f"Raw advantages range: [{data['advantages'].min():.6f}, {data['advantages'].max():.6f}]")
            print(f"Mean episodic cost: {data['mean_ep_cost']:.6f}")
        
        # Normalize advantages globally
        data['advantages'] = (data['advantages'] - data['advantages'].mean()) / (data['advantages'].std() + 1e-8)

        policy_losses, value_losses, cost_value_losses = [], [], []
        reward_losses, cost_losses = [], []
        clip_fractions, entropies = [], []
        
        dataset = TensorDataset(data['states'], data['actions'], data['log_probs_old'], 
                               data['advantages'], data['returns'], data['cost_returns'],
                               data['values_old'], data['cost_values_old'], data['cost_advantages'])
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(epochs):
            for states_b, actions_b, log_probs_old_b, advantages_b, returns_b, cost_returns_b, values_old_b, cost_values_old_b, cost_advantages_b in dataloader:
                log_probs, entropy, values, cost_values = self.actor_critic.evaluate(states_b, actions_b)
                
                # Calculate losses
                reward_loss = self._loss_pi_reward(log_probs, log_probs_old_b, advantages_b)
                cost_loss = self._loss_pi_cost(log_probs, log_probs_old_b, cost_advantages_b, data['mean_ep_cost'])
                policy_loss = reward_loss + cost_loss - self.entropy_coeff * entropy.mean()
                
                if self.debug:
                    ratios = torch.exp(log_probs - log_probs_old_b)
                    total_grad_norm = 0
                    
                    # Calculate gradients without stepping
                    self.actor_optimizer.zero_grad()
                    policy_loss.backward(retain_graph=True)
                    for p in self.actor_params:
                        if p.grad is not None:
                            total_grad_norm += p.grad.data.norm(2).item() ** 2
                    
                    print(f"Actor gradient norm: {total_grad_norm ** 0.5}")
                    print(f"  Advantages: mean={advantages_b.mean():.6f}, std={advantages_b.std():.6f}")
                    print(f"  Ratios: mean={ratios.mean():.6f}, std={ratios.std():.6f}, range=[{ratios.min():.6f}, {ratios.max():.6f}]")
                    print(f"  Reward loss: {reward_loss.item():.6f}")
                    print(f"  Cost loss: {cost_loss.item():.6f}")
                    print(f"  Total policy loss: {policy_loss.item():.6f}")
                    print(f"  Log prob diff: {(log_probs - log_probs_old_b).abs().mean():.6f}")
                
                # Actor update
                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                clip_grad_norm_(self.actor_params, self.max_grad_norm)
                self.actor_optimizer.step()

                # Critic updates with value clipping
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
                    ratios = torch.exp(log_probs - log_probs_old_b)
                    policy_losses.append(policy_loss.item())
                    reward_losses.append(reward_loss.item())
                    cost_losses.append(cost_loss.item())
                    value_losses.append(value_loss.item())
                    cost_value_losses.append(cost_value_loss.item())
                    entropies.append(entropy.mean().item())
                    clip_fractions.append((torch.abs(ratios - 1.0) > self.eps_clip).float().mean().item())

        with torch.no_grad():
            final_values = self.actor_critic.get_value(data['states']).squeeze()
            final_cost_values = self.actor_critic.get_cost_value(data['states']).squeeze()
            var_y = torch.var(data['returns'])
            explained_var_value = (1 - torch.var(data['returns'] - final_values) / (var_y + 1e-8)).item()
            var_y_cost = torch.var(data['cost_returns'])
            explained_var_cost_value = (1 - torch.var(data['cost_returns'] - final_cost_values) / (var_y_cost + 1e-8)).item()

        memory.clear_memory()

        return {
            "avg_reward_policy_loss": np.mean(reward_losses),
            "avg_cost_policy_loss": np.mean(cost_losses),
            "avg_value_loss": np.mean(value_losses),
            "avg_cost_value_loss": np.mean(cost_value_losses),
            "clip_fraction": np.mean(clip_fractions),
            "entropy": np.mean(entropies),
            "explained_var_value": explained_var_value,
            "explained_var_cost_value": explained_var_cost_value,
            "avg_rollout_cost": np.mean(data['raw_costs']),
        }

    def save_checkpoint(self, path):
        torch.save(self.actor_critic.state_dict(), path)

    def load_checkpoint(self, path):
        self.actor_critic.load_state_dict(torch.load(path, map_location=self.device))
