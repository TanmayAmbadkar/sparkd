
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict
from copy import deepcopy

from src.policies.abstract_agent import Agent
from src.vdk_shield import (
    VDK_Shield, VisualEncoder, TabularEncoder, VisualDecoder, TabularDecoder,
    compute_autoregressive_loss,
)
from src.algorithms.sac.replay_memory import ReplayMemory


class RunningMeanStd:
    """Welford's online algorithm for running mean/variance."""
    def __init__(self, shape=()):
        self.mean = torch.zeros(shape)
        self.var = torch.ones(shape)
        self.count = 1e-4

    def update(self, x: torch.Tensor):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean.to(x.device)
        total = self.count + batch_count
        
        new_mean = self.mean.to(x.device) + delta * batch_count / total
        m_a = self.var.to(x.device) * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total
        new_var = M2 / total
        
        self.mean = new_mean
        self.var = new_var
        self.count = total

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / (torch.sqrt(self.var.to(x.device)) + 1e-8)


# =========================================================================
# AVGA Agent  (Analytic Value-Gradient Agent)
# =========================================================================
class NeuralValueNet(nn.Module):
    """
    MLP value function over the Koopman latent space.
    V(z) : ℝ^{2d} → ℝ   (scalar value of concatenated [Re(z), Im(z)])

    Uses SiLU (Swish) activations — C∞ smooth, so ∇_z V is well-behaved
    everywhere (no dead zones unlike ReLU).
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        input_dim = 2 * latent_dim  # Re + Im

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Small init on final layer to prevent initial value explosions
        nn.init.uniform_(self.net[-1].weight, -1e-2, 1e-3)
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, z_re: torch.Tensor, z_im: torch.Tensor) -> torch.Tensor:
        """(B, d), (B, d) → (B, 1)"""
        z = torch.cat([z_re, z_im], dim=-1)
        return self.net(z)


# =========================================================================
# AVGA Agent  (Analytic Value-Gradient Agent)
# =========================================================================
class ALLCAgent(Agent):
    """
    Analytic Value-Gradient Agent (AVGA).

    Components:
        1. VDK Shield   – Learns latent-linear dynamics  z' = Λ⊙z + B·u
        2. NeuralValueNet – Learns V(z) via off-policy Fitted Value Iteration
        3. Gradient Solver – u* = u_max · tanh(γ/R · B^T ∇_z V)

    No SAC teacher.  Self-contained exploration → learning → control.
    """

    def __init__(self, gym_env, args):
        # ----------------------------------------------------------
        # Device
        # ----------------------------------------------------------
        if not args.cuda:
            self.device = torch.device("cpu")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.args = args
        self.observation_space = gym_env.observation_space
        self.action_space = gym_env.action_space
        self.control_dim = self.action_space.shape[0]
        self.batch_size = args.batch_size
        self.gamma = args.gamma

        # ----------------------------------------------------------
        # 1. VDK Shield (Dynamics)
        # ----------------------------------------------------------
        obs_shape = self.observation_space.shape
        is_visual = len(obs_shape) == 3

        if is_visual:
            print(f"[AVGA] Visual Encoder for shape {obs_shape}")
            encoder = VisualEncoder(latent_dim=args.red_dim, in_channels=obs_shape[0])
            decoder = VisualDecoder(latent_dim=args.red_dim, out_channels=obs_shape[0])
        else:
            print(f"[AVGA] Tabular Encoder for shape {obs_shape}")
            encoder = TabularEncoder(state_dim=obs_shape[0], latent_dim=args.red_dim)
            decoder = TabularDecoder(latent_dim=args.red_dim, state_dim=obs_shape[0])

        self.vdk = VDK_Shield(
            encoder=encoder,
            control_dim=self.control_dim,
            decoder=decoder,
            alpha=0.1,
        ).to(self.device)

        self.vdk_optimizer = optim.Adam(self.vdk.parameters(), lr=args.lr)

        # Target VDK for stable Value Learning
        self.vdk_target = deepcopy(self.vdk).to(self.device)
        for p in self.vdk_target.parameters():
            p.requires_grad_(False)
        self.tau_encoder = 0.005  # Slow update for encoder target

        # ----------------------------------------------------------
        # 2. Neural Value Network  +  Target Network
        # ----------------------------------------------------------
        value_hidden = getattr(args, "value_hidden_dim", 256)
        self.value = NeuralValueNet(args.red_dim, value_hidden).to(self.device)
        self.value_target = deepcopy(self.value).to(self.device)
        # Freeze target — updated only via soft copy
        for p in self.value_target.parameters():
            p.requires_grad_(False)

        self.value_optimizer = optim.Adam(self.value.parameters(), lr=args.lr)
        self.tau = getattr(args, "tau", 0.005)

        # ----------------------------------------------------------
        # 3. Replay Buffer  (off-policy, for value FVI)
        # ----------------------------------------------------------
        self.memory = ReplayMemory(
            capacity=args.replay_size,
            observation_space=self.observation_space,
            action_dim=self.control_dim,
            seed=args.seed,
        )

        # ----------------------------------------------------------
        # 4. On-Policy Buffer  (for VDK dynamics training)
        # ----------------------------------------------------------
        self.on_policy_buffer = []
        self.on_policy_batch_size = getattr(args, "on_policy_batch_size", 2048)
        self.initial_epochs = getattr(args, "initial_epochs", 200)
        self.on_policy_epochs = args.vll_epochs_dyn

        # ----------------------------------------------------------
        # Control Cost
        # ----------------------------------------------------------
        self.r_start = getattr(args, "r_cost", 2.0)
        self.r_end = 5.0
        self.num_steps = args.num_steps
        self.u_max = torch.tensor(
            self.action_space.high, device=self.device, dtype=torch.float32
        )



        # ----------------------------------------------------------
        # Exploration
        # ----------------------------------------------------------
        self.exploration_noise = getattr(args, "exploration_noise", 0.2)

        # ----------------------------------------------------------
        # State Normalization  (running EMA)
        # ----------------------------------------------------------
        state_dim = obs_shape[0] if not is_visual else obs_shape[0]
        self.state_mean = torch.zeros(obs_shape[0], device=self.device)
        self.state_std = torch.ones(obs_shape[0], device=self.device)

        # ----------------------------------------------------------
        # Counters
        # ----------------------------------------------------------
        self.total_steps = 0
        self.updates = 0
        self.initial_training_done = False
        
        # Reward Normalization
        self.reward_normalizer = RunningMeanStd(shape=(1,))

    @property
    def R_val(self):
        """Linearly anneal R from r_start to r_end over num_steps."""
        progress = min(1.0, self.total_steps / self.num_steps)
        return self.r_start + progress * (self.r_end - self.r_start)

    # ==================================================================
    # Analytic Gradient Solver  (with tanh squashing + drift correction)
    # ==================================================================
    def solve_analytic_u(self, z_re, z_im, use_target=False):
        """
        One-step lookahead solver:
            z_drift = Λ⊙z   (natural evolution, u=0)
            u* = u_max · tanh(γ/R · B^T ∇_z V(z_drift))

        Evaluating the gradient at the *predicted* next state accounts for
        the system's natural momentum (spectral rotation), not just the
        current position.
        """
        # Apply drift: z_drift = Λ ⊙ z  (element-wise complex multiply)
        lam_re = self.vdk.dynamics.lambda_re  # (d,)
        lam_im = self.vdk.dynamics.lambda_im  # (d,)

        # Complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        drift_re = z_re * lam_re - z_im * lam_im
        drift_im = z_re * lam_im + z_im * lam_re

        # Enable gradients for drift state
        drift_re = drift_re.detach().requires_grad_(True)
        drift_im = drift_im.detach().requires_grad_(True)

        net = self.value_target if use_target else self.value
        v = net(drift_re, drift_im)

        # ∇_z V(z_drift)
        grad_re, grad_im = torch.autograd.grad(
            v.sum(), (drift_re, drift_im), create_graph=False
        )

        # Project through B(z_drift)  (u_raw = B_re^T · g_re + B_im^T · g_im)
        B_re, B_im = self.vdk.dynamics.get_B(drift_re, drift_im) # (B, d, m)
        
        # Batch Matmul: (B, 1, d) @ (B, d, m) -> (B, 1, m)
        force_re = torch.matmul(grad_re.unsqueeze(1), B_re).squeeze(1)
        force_im = torch.matmul(grad_im.unsqueeze(1), B_im).squeeze(1)
        
        force = force_re + force_im  # (B, m)

        # Tanh squashing:  near 0 → linear (LQR),  near ±1 → saturates
        scale = self.gamma / self.R_val
        u_star = self.u_max * torch.tanh(force * scale)

        return u_star

    # ==================================================================
    # Agent Interface  (matches SAC / PPO pattern)
    # ==================================================================
    def __call__(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        return self.select_action(state, evaluate)

    def select_action(self, state, evaluate=False):
        """
        < start_steps : random actions  (exploration)
        ≥ start_steps : analytic solver + Gaussian noise
        """
        if self.total_steps < self.args.start_steps:
            return self.action_space.sample()

        state_t = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        state_t = (state_t - self.state_mean) / self.state_std

        with torch.no_grad():
            mu_re, mu_im, _, _ = self.vdk.encode(state_t)

        # solve needs grad tracking on z, not on vdk
        u_star = self.solve_analytic_u(mu_re, mu_im, use_target=False)

        # --- DEBUG BLOCK ---
        # --- DEBUG BLOCK ---
        if not evaluate and self.total_steps % 100 == 0:
            # We need to re-enable gradients just for this calculation
            with torch.enable_grad():
                # 1. Prepare fresh tensors with grad enabled
                z_re_dbg = mu_re.detach().requires_grad_(True)
                z_im_dbg = mu_im.detach().requires_grad_(True)
                
                # 2. Forward pass through the Value Net
                v_dbg = self.value(z_re_dbg, z_im_dbg)
                
                # 3. Calculate the raw gradients ∇V
                # grads will be a tuple: (grad_re, grad_im)
                grads = torch.autograd.grad(v_dbg.sum(), (z_re_dbg, z_im_dbg))
                grad_re, grad_im = grads[0], grads[1]
                
                # 4. Calculate magnitudes for logging
                grad_mag = torch.norm(grad_re) + torch.norm(grad_im)
                action_mag = u_star.norm()
                
                print(f"Step {self.total_steps:7d} | "
                    f"V: {v_dbg.mean().item():7.3f} | "
                    f"GradMag: {grad_mag.item():8.6f} | "
                    f"ActionMag: {action_mag.item():8.4f}")
        u_np = u_star.detach().cpu().numpy()[0]

        # NaN safety
        if np.isnan(u_np).any() or np.isinf(u_np).any():
            u_np = np.zeros(self.control_dim)

        # Exploration noise
        if not evaluate:
            noise = np.random.normal(0, self.exploration_noise, size=self.control_dim)
            u_np = u_np + noise

        u_np = np.clip(u_np, self.action_space.low, self.action_space.high)
        # if self.total_steps % 10 == 0:
        #     print(f"Step {self.total_steps} | Action: {u_np}")
        return u_np

    def add(self, state, action, reward, next_state, done, cost):
        if (
            not np.isfinite(state).all()
            or not np.isfinite(action).all()
            or not np.isfinite(next_state).all()
        ):
            return

        self.total_steps += 1
        self.memory.push(state, action, reward, next_state, done, cost)
        self.on_policy_buffer.append((state, action, reward, next_state, done))

    # ==================================================================
    # Train  (unified call — matches SAC / PPO)
    # ==================================================================
    def train(self) -> Dict[str, float]:
        """
        Every call:
            1. Value FVI update  (off-policy, from replay buffer)
        Periodically:
            2. VDK dynamics update  (on-policy buffer)
        """
        stats = {}

        # --- Off-Policy Value Update (FVI) ---
        off_policy_stats = self._update_value_off_policy()
        stats.update(off_policy_stats)

        # --- Joint Update (VDK + Value with GAE) ---
        joint_stats = self._update_joint()
        stats.update(joint_stats)
        
        if joint_stats:
             self.updates += 1
        
        return stats

    # ==================================================================
    # Offline Training Loop
    # ==================================================================
    def train_offline(self, steps: int = 1000, log_every: int = 100, start_step: int = 0):
        print(f"Starting Offline Training for {steps} steps (from {start_step})...")
        self.initial_training_done = True 
        
        initial_steps = getattr(self.args, "offline_initial_steps", 10000)
        dyn_interval = getattr(self.args, "offline_dyn_interval", 5000)
        initial_epochs = getattr(self.args, "initial_epochs", 200)

        # 0. Initial VAE Training (Only at absolute start)
        if start_step == 0:
            print(f"--- Initial VAE Training (Epochs: {initial_epochs} on first {initial_steps} transitions) ---")
            self._train_initial_vae(epochs=initial_epochs, sample_limit=initial_steps)

        finetune_updates = getattr(self.args, "offline_finetune_updates", 1000)

        for step in range(start_step, start_step + steps):
             stats = {}
             
             # Phase 1: Initial (Critic Only, Subset Data)
             if step < initial_steps:
                 v_stats = self._update_value_off_policy(sample_range=(0, initial_steps))
                 stats.update(v_stats)
                 
             # Phase 2: Main Loop (Periodic Coupled Updates)
             else:
                 # Check for periodic trigger
                 if (step - initial_steps) % dyn_interval == 0:
                     print(f"--- Periodic Finetuning at Step {step} ({finetune_updates} updates) ---")
                     
                     ft_v_loss = []
                     ft_d_loss = []
                     
                     for _ in range(finetune_updates):
                         # Update Both
                         d_stats = self._update_dynamics_offline(sample_range=None) # Full data
                         v_stats = self._update_value_off_policy(sample_range=None) # Full data
                         
                         if "dyn_loss_off" in d_stats: ft_d_loss.append(d_stats["dyn_loss_off"])
                         if "val_loss_off" in v_stats: ft_v_loss.append(v_stats["val_loss_off"])
                     
                     # Log average stats for this block
                     if ft_d_loss: stats["dyn_loss_off"] = np.mean(ft_d_loss)
                     if ft_v_loss: stats["val_loss_off"] = np.mean(ft_v_loss)
                     stats["finetune_block"] = 1.0
                     
                     # Add latent spread from last update if available
                     if "latent_spread_re" in d_stats:
                         stats["latent_spread_re"] = d_stats["latent_spread_re"]
                         stats["latent_spread_im"] = d_stats["latent_spread_im"]

                 else:
                     # Do nothing between intervals
                     pass
                  
             if step % log_every == 0:
                 # Filter out empty stats print if nothing happened
                 if stats:
                    print(f"Offline Step {step}: {stats}")
                 
             self.updates += 1

    def _train_initial_vae(self, epochs, sample_limit):
        """
        Pre-trains VAE on the first `sample_limit` transitions for `epochs`.
        """
        # Estimate batches per epoch
        subset_size = min(len(self.memory), sample_limit)
        batch_size = self.args.batch_size
        if subset_size < batch_size:
            print("Warning: Initial data < batch size. Training might be unstable.")
            
        n_batches = max(1, subset_size // batch_size)
        
        print(f"Training VAE for {epochs} epochs on {subset_size} samples ({n_batches} batches/epoch)...")
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for _ in range(n_batches):
                stats = self._update_dynamics_offline(sample_range=(0, sample_limit))
                epoch_loss += stats.get("dyn_loss_off", 0.0)
            
            if (epoch + 1) % 10 == 0:
                print(f"  VAE Epoch {epoch+1}/{epochs}: Loss {epoch_loss/n_batches:.4f}")
             
    def _update_dynamics_offline(self, sample_range=None):
        """
        Updates VDK dynamics using sequences sampled from ReplayMemory.
        """
        if len(self.memory) < self.batch_size:
             return {}
             
        T_horizon = getattr(self.args, "horizon", 5)
        batch_size = self.args.batch_size
        
        # Sample sequences: (B, T, D)
        # Note: ReplayMemory.sample doesn't support horizon > 1 unless configured?
        # Checked ReplayMemory: yes, supports horizon arg.
        
        try:
            states, actions, rewards, next_states, dones = self.memory.sample(batch_size, horizon=T_horizon, sample_range=sample_range)
        except ValueError:
            return {} # Not enough data
            
        # To Torch
        s_t = torch.FloatTensor(states).to(self.device)   # (B, T, D)
        u_t = torch.FloatTensor(actions).to(self.device)  # (B, T, m)
        
        # Normalize
        s_t = (s_t - self.state_mean) / self.state_std
        
        self.vdk_optimizer.zero_grad()
        loss, metrics = compute_autoregressive_loss(
            self.vdk, s_t, u_t, eps=0.5, gamma_loss=0.5, spectral_reg_weight=0.01
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.vdk.parameters(), 1.0)
        self.vdk_optimizer.step()
        
        # Soft update target
        self._soft_update(self.vdk, self.vdk_target, self.tau_encoder)

        return {
            "dyn_loss_off": loss.item(),
            "latent_spread_re": metrics.get("spread_re", 0.0).item(),
            "latent_spread_im": metrics.get("spread_im", 0.0).item(),
        }

    # ==================================================================
    # Private: Value FVI  (off-policy Bellman backup)
    # ==================================================================
    def _update_value_off_policy(self, sample_range=None) -> Dict[str, float]:
        """
        Off-Policy Value Update using Expectile Regression (IQL).
        Minimizes Expectile Loss between V(z) and target_q (r + gamma * V'(z')).
        Forces V(z) to approximate the tau-th quantile of returns (upper envelope).
        """
        # Only start off-policy updates after the VDK dynamics are initialized
        if not self.initial_training_done:
            return {}

        if len(self.memory) < self.batch_size:
            return {}

        # 1. Configuration
        tau = getattr(self.args, "offline_iql_tau", 0.7)

        # 2. Sample (Standard Batch)
        # We switch back to standard sampling (horizon=1) for IQL-style V-learning
        batch = self.memory.sample(self.batch_size, sample_range=sample_range)
        states, actions, rewards, next_states, dones = batch
        
        # 3. To Tensor
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1) # (B, 1)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)     # (B, 1)
        
        # 4. Normalize States
        states = (states - self.state_mean) / self.state_std
        next_states = (next_states - self.state_mean) / self.state_std

        # 5. Reward Normalization
        # Normalize rewards for stability
        rewards = self.reward_normalizer.normalize(rewards)
        rewards = torch.clamp(rewards, -10.0, 10.0)

        # 6. Compute Target Q (1-step Bellman using V_target)
        with torch.no_grad():
             # Encode Next State
             # We use simple mean encoding, ignoring uncertainty for the main value path as standard IQL
             mu_next_re, mu_next_im, _, _ = self.vdk.encode(next_states)
             
             # Drift Next
             lam_re = self.vdk.dynamics.lambda_re
             lam_im = self.vdk.dynamics.lambda_im
             drift_next_re = mu_next_re * lam_re - mu_next_im * lam_im
             drift_next_im = mu_next_re * lam_im + mu_next_im * lam_re
             
             # V_target(z')
             v_next = self.value_target(drift_next_re, drift_next_im)
             
             # Target Q = r + gamma * V_target(s')
             target_q = rewards + self.gamma * (1 - dones) * v_next

        # 7. Compute Prediction V(z)
        # Encode Current State (detach, no gradients to encoder)
        with torch.no_grad():
             mu_curr_re, mu_curr_im, _, _ = self.vdk.encode(states)
             drift_curr_re = mu_curr_re * lam_re - mu_curr_im * lam_im
             drift_curr_im = mu_curr_re * lam_im + mu_curr_im * lam_re
             
        # V(z) - with gradients
        v_pred = self.value(drift_curr_re, drift_curr_im)

        # 8. Expectile Loss (IQL Logic)
        # L2_tau(diff) = |tau - I(diff < 0)| * diff^2
        diff = target_q - v_pred
        weight = torch.where(diff > 0, tau, 1.0 - tau)
        loss = (weight * (diff**2)).mean()

        self.value_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), 1.0)
        self.value_optimizer.step()

        # 9. Soft Update Target
        self._soft_update(self.value, self.value_target, self.tau)

        return {"val_loss_off": loss.item()}


    @staticmethod
    def _soft_update(source, target, tau):
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)

    # ==================================================================
    # Private: VDK Dynamics Update  (phased, on-policy)
    # ==================================================================
    # ==================================================================
    # Joint Update (VDK + Value GAE)
    # ==================================================================
    def _update_joint(self) -> Dict[str, float]:
        """
        1. Train VDK on on-policy buffer.
        2. Re-encode buffer -> z, z_next
        3. Compute GAE Returns.
        4. Train ValueNet on (z, Returns).
        """
        if self.total_steps < self.args.start_steps:
             return {}
 
        is_initial = not self.initial_training_done
        is_periodic = len(self.on_policy_buffer) >= self.on_policy_batch_size
 
        if not is_initial and not is_periodic:
             return {}
 
        label = "INITIAL" if is_initial else "JOINT"
        N = len(self.on_policy_buffer)
        print(f"\n[AVGA] {label} UPDATE on {N} samples...")
 
        # --- Prepare data ---
        T_horizon = getattr(self.args, "horizon", 5)
        states_np = np.array([x[0] for x in self.on_policy_buffer])
        actions_np = np.array([x[1] for x in self.on_policy_buffer])
        rewards_np = np.array([x[2] for x in self.on_policy_buffer])
        next_states_np = np.array([x[3] for x in self.on_policy_buffer])
        dones_np = np.array([x[4] for x in self.on_policy_buffer])
 
        # --- State normalization ---
        full_states = torch.FloatTensor(states_np).to(self.device)
        self.state_mean = full_states.mean(dim=0)
        self.state_std = full_states.std(dim=0) + 1e-6
        if is_initial:
            print(f"State Mean: {self.state_mean[:4]}... | Std: {self.state_std[:4]}...")
        self.state_std[self.state_std < 1e-6] = 1.0
 
        # --- 1. Train VDK ---
        valid_starts = []
        for i in range(N - T_horizon + 1):
             if not np.any(dones_np[i : i + T_horizon - 1]):
                 valid_starts.append(i)
 
        final_vdk_loss = 0.0
        if valid_starts:
            epochs = self.initial_epochs if is_initial else self.args.vll_finetune_epochs_dyn
            batch_size = len(valid_starts) if is_initial else self.on_policy_batch_size
            
            for epoch in range(epochs):
                idx = np.random.choice(valid_starts, size=min(batch_size, len(valid_starts)), replace=True)
                s_batch = np.array([states_np[i : i + T_horizon] for i in idx])
                u_batch = np.array([actions_np[i : i + T_horizon] for i in idx])
 
                s_t = torch.FloatTensor(s_batch).to(self.device)
                u_t = torch.FloatTensor(u_batch).to(self.device)
                s_t = (s_t - self.state_mean) / self.state_std
 
                self.vdk_optimizer.zero_grad()
                loss, metrics = compute_autoregressive_loss(
                    self.vdk, s_t, u_t, eps=0.5, gamma_loss=0.5, spectral_reg_weight=0.01
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.vdk.parameters(), 1.0)
                self.vdk_optimizer.step()
                final_vdk_loss = loss.item()
                
                # Check collapse
                if epoch == epochs - 1:
                     with torch.no_grad():
                         mu0_re, mu0_im, _, _ = self.vdk.encode(s_t[:, 0, :])
                         spread = mu0_re.var(dim=0).mean().item()
                     if spread < 1e-4:
                         print("WARNING: Latent Collapse!")
 
        # --- 2. Train ValueNet (On-Policy GAE) ---
        # Re-encode FULL buffer with updated VDK to get fresh latents
        # We process in chunks to avoid OOM if buffer is huge
        
        # Normalize states
        full_states_norm = (torch.FloatTensor(states_np).to(self.device) - self.state_mean) / self.state_std
        full_next_norm = (torch.FloatTensor(next_states_np).to(self.device) - self.state_mean) / self.state_std
        
        with torch.no_grad():
             # Encode current s
             mu_re, mu_im, _, _ = self.vdk.encode(full_states_norm)
             # Encode next s'
             mu_next_re, mu_next_im, _, _ = self.vdk.encode(full_next_norm)
             
             # Apply Drift Correction (evaluate V at drifted z)
             lam_re = self.vdk.dynamics.lambda_re
             lam_im = self.vdk.dynamics.lambda_im
             
             drift_re = mu_re * lam_re - mu_im * lam_im
             drift_im = mu_re * lam_im + mu_im * lam_re
             
             drift_next_re = mu_next_re * lam_re - mu_next_im * lam_im
             drift_next_im = mu_next_re * lam_im + mu_next_im * lam_re
             
             # Compute V(z) and V(z') using CURRENT ValueNet (bootstrap)
             values = self.value(drift_re, drift_im).squeeze(-1)       # (N,)
             next_values = self.value(drift_next_re, drift_next_im).squeeze(-1) # (N,)
        
        # Rewards
        r_tensor = torch.FloatTensor(rewards_np).to(self.device)
        d_tensor = torch.FloatTensor(dones_np).to(self.device)
        
        # Normalize Rewards
        self.reward_normalizer.update(r_tensor.unsqueeze(1))
        r_norm = self.reward_normalizer.normalize(r_tensor.unsqueeze(1)).squeeze(1)
        r_norm = torch.clamp(r_norm, -10.0, 10.0)
        
        # Calculate GAE
        gae_lam = getattr(self.args, "gae_lambda", 0.95)
        advantages = torch.zeros_like(r_norm)
        last_gae = 0.0
        
        # Loop backwards
        # Note: self.on_policy_buffer is sequential (s, s', r, d)
        for t in reversed(range(N)):
            if t == N - 1:
                next_non_terminal = 1.0 - d_tensor[t]
                next_val = next_values[t] # Bootstrap from last step value
            else:
                next_non_terminal = 1.0 - d_tensor[t]
                next_val = values[t+1] # Bootstrap from next step in buffer (if not done)
                # If done, next_non_terminal handles it.
                # Ideally next_val should be V(s_{t+1}).
                # But wait, s_{t+1} IS stored as next_s in transition t.
                # So next_values[t] is correct: V(next_s_t).
                next_val = next_values[t]

            delta = r_norm[t] + self.gamma * next_non_terminal * next_val - values[t]
            last_gae = delta + self.gamma * gae_lam * next_non_terminal * last_gae
            advantages[t] = last_gae
            
        returns = advantages + values
        
        # Train ValueNet on (z, returns)
        # We use strict MSE on returns
        v_epochs = getattr(self.args, "value_epochs", 10)
        v_batch_size = getattr(self.args, "value_batch_size", 64)
        
        final_val_loss = 0.0
        indices = np.arange(N)
        
        for _ in range(v_epochs):
             np.random.shuffle(indices)
             for i in range(0, N, v_batch_size):
                 batch_idx = indices[i : i + v_batch_size]
                 
                 # Inputs: drift_re[batch_idx], drift_im[batch_idx]
                 # Targets: returns[batch_idx]
                 
                 z_re_batch = drift_re[batch_idx]
                 z_im_batch = drift_im[batch_idx]
                 target_batch = returns[batch_idx].unsqueeze(1)
                 
                 # Forward
                 v_pred = self.value(z_re_batch.detach(), z_im_batch.detach()) # Detach from VDK graph? YES. VDK fixed here.
                 
                 loss = F.mse_loss(v_pred, target_batch)
                 
                 self.value_optimizer.zero_grad()
                 loss.backward()
                 torch.nn.utils.clip_grad_norm_(self.value.parameters(), 1.0)
                 self.value_optimizer.step()
                 final_val_loss = loss.item()

        # Update VDK Target (Soft Update after joint training)
        self._soft_update(self.vdk, self.vdk_target, self.tau_encoder)
        
        # Update Value Target (Soft Update)
        self._soft_update(self.value, self.value_target, self.tau)

        # Cleanup
        self.on_policy_buffer = []
        if is_initial:
            self.initial_training_done = True
            
        return {"dyn_loss": final_vdk_loss, "val_loss": final_val_loss}

    # ==================================================================
    # Checkpointing
    # ==================================================================
    def save_checkpoint(self, path):
        torch.save(
            {
                "vdk": self.vdk.state_dict(),
                "vdk_target": self.vdk_target.state_dict(),
                "value": self.value.state_dict(),
                "value_target": self.value_target.state_dict(),
                "state_mean": self.state_mean,
                "state_std": self.state_std,
            },
            path,
        )

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.vdk.load_state_dict(ckpt["vdk"])
        if "vdk_target" in ckpt:
            self.vdk_target.load_state_dict(ckpt["vdk_target"])
        else:
            self.vdk_target.load_state_dict(ckpt["vdk"])
        self.value.load_state_dict(ckpt["value"])
        self.value_target.load_state_dict(ckpt["value_target"])
        if "state_mean" in ckpt:
            self.state_mean = ckpt["state_mean"].to(self.device)
            self.state_std = ckpt["state_std"].to(self.device)
