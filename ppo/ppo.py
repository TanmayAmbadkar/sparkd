import torch
import torch.nn.functional as F
from torch.optim import Adam
from ppo.model import ActorCritic
import numpy as np
from sklearn.metrics import explained_variance_score
from ppo.utils import RunningMeanStd

class PPO:
    def __init__(self, obs_dim, action_space, args):
        self.gamma = 0.995
        self.lam = 0.95  # GAE lambda
        self.eps_clip = 0.2
        self.value_coeff = 0.5
        self.entropy_coeff = 0.0
        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.actor_critic = ActorCritic(obs_dim, action_space, args.hidden_size).to(self.device)
        self.optimizer = Adam(self.actor_critic.parameters(), lr=args.lr)
        self.action_space = action_space

        # Normalization stats are kept as numpy arrays on the CPU
        self.reward_rms = RunningMeanStd(reward_size=1)
        self.state_rms = RunningMeanStd(reward_size=obs_dim)

    def select_action(self, state):
        # Normalization happens on CPU with numpy
        state_norm = self.state_rms.normalize(state)
        # Convert to tensor and move to the correct device for the model
        state_tensor = torch.Tensor(state_norm).to(self.device).unsqueeze(0)

        with torch.no_grad():
            action, log_prob = self.actor_critic.act(state_tensor)

        # Move action and log_prob back to CPU to be returned as numpy arrays
        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0]

    @torch.no_grad()
    def compute_advantages_and_returns(self, memory):
        """
        Compute returns and advantages for a rollout using GAE.
        This function now correctly handles device placement for all tensors.
        """
        # --- 1. Data Collection and Normalization (on CPU) ---
        states = memory.states[:memory.size]
        self.state_rms.update(states)
        states_norm = self.state_rms.normalize(states)

        rewards = memory.rewards[:memory.size]
        self.reward_rms.update(rewards)
        rewards_norm = self.reward_rms.normalize(rewards)

        dones = memory.dones[:memory.size]
        next_states = memory.next_states[:memory.size]
        next_states_norm = self.state_rms.normalize(next_states)
        actions = memory.actions[:memory.size]

        # --- 2. Convert to Tensors and Move to Target Device ---
        state_tensor = torch.tensor(states_norm, dtype=torch.float32).to(self.device)
        next_state_tensor = torch.tensor(next_states_norm, dtype=torch.float32).to(self.device)
        action_tensor = torch.tensor(actions, dtype=torch.float32).to(self.device)
        reward_tensor = torch.tensor(rewards_norm, dtype=torch.float32).to(self.device)
        done_tensor = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # --- 3. GAE Calculation (on Target Device) ---
        values = self.actor_critic.get_value(state_tensor).squeeze()
        next_values = self.actor_critic.get_value(next_state_tensor).squeeze()

        N = len(rewards)
        # Create returns and advantages tensors directly on the target device
        returns = torch.zeros(N, device=self.device)
        advantages = torch.zeros(N, device=self.device)
        gae = 0.0

        for t in reversed(range(N)):
            mask = 1.0 - done_tensor[t]
            delta = reward_tensor[t] + self.gamma * next_values[t] * mask - values[t]
            gae = delta + self.gamma * self.lam * mask * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
        
        # Get old log probabilities for the PPO update
        log_probs_old = self.actor_critic.get_log_prob(state_tensor, action_tensor)

        return state_tensor, action_tensor, log_probs_old, returns, advantages

    def update_parameters(self, memory, epochs, batch_size):
        states, actions, log_probs_old, returns, advantages = self.compute_advantages_and_returns(memory)

        # Normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- FIX: Initialize accumulators BEFORE the epoch loop ---
        total_policy_loss, total_value_loss, total_entropy_loss = 0.0, 0.0, 0.0
        total_loss, total_clip_frac, total_kl_div, total_explained_var = 0.0, 0.0, 0.0, 0.0
        num_updates = 0

        for _ in range(epochs):
            for idx in range(0, len(states), batch_size):
                batch_slice = slice(idx, idx + batch_size)

                # All tensors are already on the correct device
                log_probs, entropy, values = self.actor_critic.evaluate(
                    states[batch_slice], actions[batch_slice]
                )

                ratios = torch.exp(log_probs - log_probs_old[batch_slice])
                
                # Policy Loss (Clipped Surrogate Objective)
                norm_advantages = advantages[batch_slice].unsqueeze(1)
                surr1 = ratios * norm_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * norm_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value Loss
                value_loss = F.mse_loss(values, returns[batch_slice].unsqueeze(1))
                
                # Entropy Loss
                entropy_loss = entropy.mean()

                # Total Loss
                loss = policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy_loss

                # --- Optimization Step ---
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=0.5)
                self.optimizer.step()

                # --- Logging Metrics ---
                clipped_mask = (ratios > (1 + self.eps_clip)) | (ratios < (1 - self.eps_clip))
                total_clip_frac += torch.mean(clipped_mask.float()).item()
                total_kl_div += torch.mean(log_probs_old[batch_slice] - log_probs).item()
                
                explained_variance = explained_variance_score(
                    returns[batch_slice].cpu().numpy(),
                    values.detach().cpu().numpy(),
                )
                total_explained_var += explained_variance

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_loss += loss.item()
                num_updates += 1

        memory.clear_memory()

        # Compute average metrics over all mini-batch updates across all epochs
        return {
            "avg_policy_loss": total_policy_loss / num_updates,
            "avg_value_loss": total_value_loss / num_updates,
            "avg_entropy_loss": total_entropy_loss / num_updates,
            "avg_total_loss": total_loss / num_updates,
            "avg_clip_fraction": total_clip_frac / num_updates,
            "avg_kl_divergence": total_kl_div / num_updates,
            "avg_explained_variance": total_explained_var / num_updates,
        }

    def save_checkpoint(self, path):
        torch.save(self.actor_critic.state_dict(), path)

    def load_checkpoint(self, path):
        self.actor_critic.load_state_dict(torch.load(path, map_location=self.device))

