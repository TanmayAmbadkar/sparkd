
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple, Dict

from src.policies.abstract_agent import Agent
from src.policies.sac_agent import SACPolicy
from src.vdk_shield import VDK_Shield, QuadraticStudent, VisualEncoder, TabularEncoder, VisualDecoder, TabularDecoder, stage1_loss, compute_autoregressive_loss


class RunningMeanStd:
    """Welford's online algorithm for running mean/variance."""
    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, x: torch.Tensor):
        batch_mean = x.mean().item()
        batch_var = x.var().item()
        batch_count = x.numel()
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        self.var = (self.var * self.count + batch_var * batch_count +
                    delta**2 * self.count * batch_count / total) / total
        self.count = total

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.var**0.5 + 1e-8)


class ALLCAgent(Agent):
    """
    Analytic Latent-Linear Controller (ALL-C).
    
    Components:
    1. VDK Shield: Learns Latent Dynamics z' = Az + Bu
    2. Teacher (SAC): Learns Q(s, a) to guide the student.
    3. Student (Quadratic): Learns V(z) = z^T P z + w^T z + beta by distilling Q.
    
    Action Selection:
    - Analytic solution maximizing V(z_next) - u^T R u
    """
    
    def __init__(self, gym_env, args, sac_args):
        # Strict device logic: if cuda=False, force CPU.
        if not args.cuda:
            self.device = torch.device("cpu")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.args = args
        self.observation_space = gym_env.observation_space
        self.action_space = gym_env.action_space
        self.control_dim = self.action_space.shape[0]
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        
        # --- 1. Teacher (SAC) ---
        self.teacher = SACPolicy(gym_env, args.replay_size, args.seed, args.batch_size, sac_args)
        self.memory = self.teacher.memory
        
        # --- 2. VDK Shield (Dynamics) ---
        obs_shape = self.observation_space.shape
        is_visual = len(obs_shape) == 3
        
        if is_visual:
            print(f"[ALL-C] Initializing Visual Encoder for shape {obs_shape}")
            encoder = VisualEncoder(latent_dim=args.red_dim, in_channels=obs_shape[0])
            decoder = VisualDecoder(latent_dim=args.red_dim, out_channels=obs_shape[0])
        else:
            print(f"[ALL-C] Initializing Tabular Encoder for shape {obs_shape}")
            encoder = TabularEncoder(state_dim=obs_shape[0], latent_dim=args.red_dim)
            decoder = TabularDecoder(latent_dim=args.red_dim, state_dim=obs_shape[0])
            
        self.vdk = VDK_Shield(
            encoder=encoder,
            control_dim=self.control_dim,
            decoder=decoder,
            alpha=0.1
        ).to(self.device)
        
        # Explicitly move components to device
        self.teacher.agent.critic.to(self.device)
        self.teacher.agent.critic_target.to(self.device)
        self.teacher.agent.policy.to(self.device)

        self.vdk_optimizer = optim.Adam(self.vdk.parameters(), lr=args.lr)
        self.finetune_lr = getattr(args, 'finetune_lr', 1e-5)
        self.vdk_finetune_optimizer = optim.Adam(self.vdk.parameters(), lr=self.finetune_lr)
        
        # --- 3. Student (Quadratic Value) ---
        self.student = QuadraticStudent(latent_dim=args.red_dim).to(self.device)
        self.student_optimizer = optim.Adam(self.student.parameters(), lr=3e-4)
        
        # On-Policy Buffer
        self.on_policy_buffer = []
        self.on_policy_batch_size = getattr(args, 'on_policy_batch_size', 2048)
        self.initial_epochs = getattr(args, 'initial_epochs', 200)
        self.on_policy_epochs = args.vll_epochs_dyn
        
        # Control Cost Matrix R (diagonal)
        self.R_val = 0.1
        self.R_tensor = torch.eye(self.control_dim, device=self.device) * self.R_val
        
        # Env Bounds
        self.u_min = torch.tensor(self.action_space.low, device=self.device, dtype=torch.float32)
        self.u_max = torch.tensor(self.action_space.high, device=self.device, dtype=torch.float32)

        # Training State
        self.total_steps = 0
        self.updates = 0
        self.initial_training_done = False
        
        # Data Normalization Stats (Initialized to Identity)
        self.state_mean = torch.zeros(self.observation_space.shape[0], device=self.device)
        self.state_std  = torch.ones(self.observation_space.shape[0], device=self.device)
        
        # Return normalization for Student targets
        self.return_normalizer = RunningMeanStd()

    # ------------------------------------------------------------------
    # Analytic Solver
    # ------------------------------------------------------------------
    def solve_analytic_u(self, z_re, z_im):
        A = self.vdk.dynamics.get_real_A_matrix()
        B = self.vdk.dynamics.get_real_B_matrix()
        P = self.student.get_P_symmetric()
        w = self.student.w
        
        z_tilde = torch.cat([z_re, z_im], dim=-1).unsqueeze(-1)
        
        M = self.R_tensor - (B.T @ P @ B)
        term1 = B.T @ P @ A
        term2 = 0.5 * (B.T @ w)
        b = (term1 @ z_tilde) + term2
        
        M_batch = M.unsqueeze(0).expand(z_tilde.size(0), -1, -1)
        
        try:
            if self.device.type == 'mps':
                 M_batch_cpu = M_batch.cpu()
                 b_cpu = b.cpu()
                 u_star_cpu = torch.linalg.solve(M_batch_cpu, b_cpu)
                 u_star = u_star_cpu.to(self.device)
            else:
                 u_star = torch.linalg.solve(M_batch, b)
        except RuntimeError:
            u_star = torch.zeros_like(b)
        
        return u_star.squeeze(-1)

    # ------------------------------------------------------------------
    # Agent Interface (matches SAC/PPO)
    # ------------------------------------------------------------------
    def __call__(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        return self.select_action(state, evaluate)

    def select_action(self, state, evaluate=False):
        """
        Phase 1 (< start_steps): Use SAC Teacher.
        Phase 2 (>= start_steps): Use Analytic Solver.
        """
        if self.total_steps < self.args.start_steps:
            return self.teacher(state, evaluate)
        
        state_t = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        state_t = (state_t - self.state_mean) / self.state_std
        
        with torch.no_grad():
            mu_re, mu_im, _, _ = self.vdk.encode(state_t)
            z_re, z_im = mu_re, mu_im
            
            u_star = self.solve_analytic_u(z_re, z_im)
            u_np = u_star.cpu().numpy()[0]
            
            if np.isnan(u_np).any() or np.isinf(u_np).any():
                u_np = np.zeros(self.control_dim)
            
            if not evaluate:
                noise = np.random.normal(0, 0.1, size=self.control_dim)
                u_np = u_np + noise
                
            u_np = np.clip(u_np, self.action_space.low, self.action_space.high)
            
        return u_np

    def add(self, state, action, reward, next_state, done, cost):
        if not np.isfinite(state).all() or not np.isfinite(action).all() or not np.isfinite(next_state).all():
             return
        
        self.total_steps += 1
        self.teacher.add(state, action, reward, next_state, done, cost)
        self.on_policy_buffer.append((state, action, reward, next_state, done))

    def train(self) -> Dict[str, float]:
        """
        Single unified training call (matches SAC/PPO pattern).
        
        1. Always: Teacher (SAC) update (off-policy).
        2. Phased: VDK + Student update (on-policy batch).
        
        Returns a flat dict of all losses.
        """
        stats = {}
        
        # --- 1. Teacher (SAC) Update ---
        sac_stats = self.teacher.train()
        self.updates += 1
        stats['critic_1'] = sac_stats[0]
        stats['critic_2'] = sac_stats[1]
        stats['policy'] = sac_stats[2]
        stats['entropy'] = sac_stats[3]
        stats['alpha'] = sac_stats[4]
        
        # --- 2. VDK/Student Update (phased) ---
        vdk_stats = self._update_vdk_student()
        stats.update(vdk_stats)
        
        return stats

    # ------------------------------------------------------------------
    # Private: VDK/Student Update
    # ------------------------------------------------------------------
    def _update_vdk_student(self) -> Dict[str, float]:
        """
        Phased VDK and Student training.
        
        Phase 1: Initial training once we hit start_steps.
        Phase 2: Periodic updates every on_policy_batch_size steps.
        """
        if self.total_steps < self.args.start_steps:
            return {}
        
        is_initial_train = (not self.initial_training_done)
        is_periodic_train = (len(self.on_policy_buffer) >= self.on_policy_batch_size)
        
        if is_initial_train:
             print(f"[ALL-C] Phase 1 Complete. Running INITIAL TRAINING on {len(self.on_policy_buffer)} samples...")
        elif is_periodic_train:
             print(f"[ALL-C] Phase 2: Running Periodic Update on {len(self.on_policy_buffer)} samples...")
        else:
            return {}
        
        # --- Prepare Data ---
        T_horizon = getattr(self.args, 'horizon', 5)
        
        states_np = np.array([x[0] for x in self.on_policy_buffer])
        actions_np = np.array([x[1] for x in self.on_policy_buffer])
        dones_np = np.array([x[4] for x in self.on_policy_buffer])
        rewards_np = np.array([x[2] for x in self.on_policy_buffer])
        next_states_np = np.array([x[3] for x in self.on_policy_buffer])
        
        # Normalization: compute on first training, EMA update on periodic
        full_states = torch.FloatTensor(states_np).to(self.device)
        batch_mean = full_states.mean(dim=0)
        batch_std  = full_states.std(dim=0) + 1e-6
        
        if is_initial_train:
             print("[ALL-C] Computing State Normalization Stats...")
             self.state_mean = batch_mean
             self.state_std  = batch_std
             print(f"[ALL-C] State Mean: {self.state_mean[:4]}... | Std: {self.state_std[:4]}...")
        else:
             # EMA update for Phase 2
             alpha = 0.01
             self.state_mean = (1 - alpha) * self.state_mean + alpha * batch_mean
             self.state_std  = (1 - alpha) * self.state_std  + alpha * batch_std
        
        if (self.state_std < 1e-6).any():
             self.state_std[self.state_std < 1e-6] = 1.0
             
        # Find valid temporal sequences
        valid_starts = []
        N = len(self.on_policy_buffer)
        for i in range(N - T_horizon + 1):
             segment_dones = dones_np[i : i+T_horizon-1]
             if not np.any(segment_dones):
                  valid_starts.append(i)
        
        if not valid_starts:
             return {}

        # --- A. Train VDK (Dynamics) ---
        epochs = self.on_policy_epochs if is_initial_train else self.args.vll_finetune_epochs_dyn
        batch_size = len(valid_starts) if is_initial_train else self.on_policy_batch_size
        vdk_opt = self.vdk_optimizer
        final_vdk_loss = 0
        
        for epoch in range(epochs):
             batch_indices = np.random.choice(valid_starts, size=min(batch_size, len(valid_starts)), replace=True)
             
             s_batch_list = []
             u_batch_list = []
             for idx in batch_indices:
                  s_batch_list.append(states_np[idx : idx+T_horizon])
                  u_batch_list.append(actions_np[idx : idx+T_horizon])
                  
             s_batch = torch.FloatTensor(np.array(s_batch_list)).to(self.device)
             u_batch = torch.FloatTensor(np.array(u_batch_list)).to(self.device)
             s_batch = (s_batch - self.state_mean) / self.state_std

             vdk_opt.zero_grad()
             loss_total, metrics = compute_autoregressive_loss(
                 self.vdk, s_batch, u_batch, eps=0.5, gamma_loss=0.5, spectral_reg_weight=0.01
             )
             loss_total.backward()
             torch.nn.utils.clip_grad_norm_(self.vdk.parameters(), 1.0)
             vdk_opt.step()
             
             final_vdk_loss = loss_total.item()
             loss_spec = metrics["spec"]
             
             with torch.no_grad():
                 mu0_re, mu0_im, _, _ = self.vdk.encode(s_batch[:, 0, :])
                 spread_re = mu0_re.var(dim=0).mean().item()
                 spread_im = mu0_im.var(dim=0).mean().item()
                 mean_spread = spread_re + spread_im
                 
             print(f"Epoch {epoch+1}: Loss = {final_vdk_loss:.4f} (Spec: {loss_spec.item():.4f}) | LatentSpread: {mean_spread:.2e}")
             
             if mean_spread < 1e-4:
                 print("WARNING: Latent Space Collapse Detected! (Spread < 1e-4)")

        # --- B. Train Student (Value) with Monte Carlo Returns ---
        # Compute discounted returns via backward pass through episodes
        mc_returns = np.zeros(N, dtype=np.float32)
        G = 0.0
        for i in reversed(range(N)):
            if dones_np[i]:
                G = 0.0
            G = rewards_np[i] + self.gamma * G
            mc_returns[i] = G
        
        s_t = torch.FloatTensor(states_np).to(self.device)
        s_t = (s_t - self.state_mean) / self.state_std
        returns_t = torch.FloatTensor(mc_returns).to(self.device).unsqueeze(1)
        
        final_student_loss = 0
        
        print("\n=== Stage 2: Training Student (Value, MC Returns) ===")
        for epoch in range(epochs):
            self.student_optimizer.zero_grad()
            with torch.no_grad():
                 mu_re, mu_im, _, _ = self.vdk.encode(s_t)
                 z_tilde = torch.cat([mu_re, mu_im], dim=-1)
            
            v_pred = self.student(z_tilde.detach())
            student_loss = F.mse_loss(v_pred, returns_t)
            student_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            self.student_optimizer.step()
            final_student_loss = student_loss.item()
            
            print(f"Epoch {epoch+1}: Loss = {final_student_loss:.4f}")
        
        # Clear Buffer & Update State
        self.on_policy_buffer = []
        if is_initial_train:
             self.initial_training_done = True
        
        return {'vdk_loss': final_vdk_loss, 'student_loss': final_student_loss}

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def save_checkpoint(self, path):
        torch.save({
            'vdk': self.vdk.state_dict(),
            'student': self.student.state_dict(),
            'teacher_policy': self.teacher.agent.policy.state_dict(),
            'teacher_critic': self.teacher.agent.critic.state_dict(),
            'teacher_critic_target': self.teacher.agent.critic_target.state_dict(),
            'state_mean': self.state_mean,
            'state_std': self.state_std,
        }, path)
        
    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.vdk.load_state_dict(ckpt['vdk'])
        self.student.load_state_dict(ckpt['student'])
        if 'teacher_policy' in ckpt:
            self.teacher.agent.policy.load_state_dict(ckpt['teacher_policy'])
            self.teacher.agent.critic.load_state_dict(ckpt['teacher_critic'])
            self.teacher.agent.critic_target.load_state_dict(ckpt['teacher_critic_target'])
        if 'state_mean' in ckpt:
            self.state_mean = ckpt['state_mean'].to(self.device)
            self.state_std = ckpt['state_std'].to(self.device)
