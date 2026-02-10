
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple, Dict

from src.policies.abstract_agent import Agent
from src.policies.sac_agent import SACPolicy
from src.vdk_shield import VDK_Shield, QuadraticStudent, VisualEncoder, TabularEncoder, VisualDecoder, TabularDecoder, stage1_loss, compute_autoregressive_loss
from tqdm import tqdm

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
        
        # --- 1. Teacher (SAC) ---
        self.teacher = SACPolicy(gym_env, args.replay_size, args.seed, args.batch_size, sac_args)
        self.memory = self.teacher.memory
        
        # --- 2. VDK Shield (Dynamics) ---
        # Determine Encoder
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
        
        # Explicitly move components to device if not handled by to()
        self.teacher.agent.critic.to(self.device)
        self.teacher.agent.critic_target.to(self.device)
        # Policy is not used for action selection but updated
        self.teacher.agent.policy.to(self.device)

        self.vdk_optimizer = optim.Adam(self.vdk.parameters(), lr=args.lr) # Re-use args.lr or config?
        
        
        # --- 3. Student (Quadratic Value) ---
        self.student = QuadraticStudent(latent_dim=args.red_dim).to(self.device)
        self.student_optimizer = optim.Adam(self.student.parameters(), lr=3e-4)
        
        # On-Policy Buffer
        self.on_policy_buffer = []
        self.on_policy_batch_size = 2048
        self.on_policy_epochs = self.args.vll_epochs_dyn # K epochs
        
        # Control Cost Matrix R (diagonal)
        self.R_val = 0.1
        self.R_tensor = torch.eye(self.control_dim, device=self.device) * self.R_val
        
        # Env Bounds
        self.u_min = torch.tensor(self.action_space.low, device=self.device, dtype=torch.float32)
        self.u_max = torch.tensor(self.action_space.high, device=self.device, dtype=torch.float32)

        # Training State
        self.total_steps = 0
        self.initial_epochs = 200
        self.initial_training_done = False
        
        # Data Normalization Stats (Initialized to Identity)
        self.state_mean = torch.zeros(self.observation_space.shape[0], device=self.device)
        self.state_std  = torch.ones(self.observation_space.shape[0], device=self.device)


    # ------------------------------------------------------------------
    # Analytic Solver
    # ------------------------------------------------------------------
    def solve_analytic_u(self, z_re, z_im):
        # ... (unchanged) ...
        # Get Real Matrices
        A = self.vdk.dynamics.get_real_A_matrix() # (2d, 2d)
        B = self.vdk.dynamics.get_real_B_matrix() # (2d, m)
        P = self.student.get_P_symmetric()        # (2d, 2d)
        w = self.student.w                        # (2d, 1)
        
        # Construct z_tilde
        z_tilde = torch.cat([z_re, z_im], dim=-1).unsqueeze(-1) # (B, 2d, 1)
        
        # 1. Compute Matrix M = (R - B^T P B)
        M = self.R_tensor - (B.T @ P @ B)
        
        # 2. Compute Vector b = B^T P A z + 0.5 B^T w
        term1 = B.T @ P @ A     # (m, 2d)
        term2 = 0.5 * (B.T @ w) # (m, 1)
        
        b = (term1 @ z_tilde) + term2
        
        # 3. Solve u = M^{-1} b
        # Expand M to (1, m, m)
        M_batch = M.unsqueeze(0).expand(z_tilde.size(0), -1, -1)
        
        # MPS Fallback
        try:
            if self.device.type == 'mps':
                 M_batch_cpu = M_batch.cpu()
                 b_cpu = b.cpu()
                 u_star_cpu = torch.linalg.solve(M_batch_cpu, b_cpu)
                 u_star = u_star_cpu.to(self.device)
            else:
                 u_star = torch.linalg.solve(M_batch, b) # (B, m, 1)
        except RuntimeError as e:
            # print("ERROR IN SOLVE ANALYTIC U")
            # print(f"[ALL-C] Linear Solve Error: {e}. Returning zeros.")
            u_star = torch.zeros_like(b)
        
        return u_star.squeeze(-1) # (B, m)

    def __call__(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        return self.select_action(state, evaluate)

    def select_action(self, state, evaluate=False):
        """
        Action Selection:
        Phase 1 (< start_steps): Use SAC Teacher.
        Phase 2 (>= start_steps): Use Analytic Solver.
        """
        # --- Phase 1: Pretraining ---
        if self.total_steps < self.args.start_steps:
            return self.teacher(state, evaluate)
        
        # --- Phase 2: Analytic Control ---
        state_t = torch.FloatTensor(state).to(self.device).unsqueeze(0) # (1, obs_dim)
        
        # Normalize State
        state_t = (state_t - self.state_mean) / self.state_std
        
        # 1. Encode
        
        # 1. Encode
        with torch.no_grad():
            mu_re, mu_im, logvar_re, logvar_im = self.vdk.encode(state_t)
            # Use Mean for control stability
            z_re, z_im = mu_re, mu_im
            
            # 2. Solve Analytic
            u_star = self.solve_analytic_u(z_re, z_im)
            
            u_np = u_star.cpu().numpy()[0]
            
            # CHECK FOR NANs
            if np.isnan(u_np).any() or np.isinf(u_np).any():
                # print("[ALL-C] Warning: Analytic Solver returned NaN/Inf. Fallback to random/zero.")
                u_np = np.zeros(self.control_dim)
            
            # 3. Exploration (if training)
            if not evaluate:
                # Add noise
                noise = np.random.normal(0, 0.1, size=self.control_dim)
                u_np = u_np + noise
                
            # 4. Clip
            u_np = np.clip(u_np, self.action_space.low, self.action_space.high)
            
        return u_np

    def add(self, state, action, reward, next_state, done, cost):
        # NaN / Inf Check
        if not np.isfinite(state).all() or not np.isfinite(action).all() or not np.isfinite(next_state).all():
             # print("[ALL-C] Warning: Detected NaN/Inf in transition. Skipping add.")
             return
        
        self.total_steps += 1

        # Add to Teacher Memory (Off-Policy SAC)
        self.teacher.add(state, action, reward, next_state, done, cost)
        
        # Add to On-Policy Buffer (VDK/Student)
        self.on_policy_buffer.append((state, action, reward, next_state, done))
        
    def save_checkpoint(self, path):
        # ... (Same logic as before) ...
        torch.save({
            'vdk': self.vdk.state_dict(),
            'student': self.student.state_dict(),
            'teacher_policy': self.teacher.agent.policy.state_dict(),
            'teacher_critic': self.teacher.agent.critic.state_dict(),
            'teacher_critic_target': self.teacher.agent.critic_target.state_dict(),
        }, path)
        
    def load_checkpoint(self, path):
        # ... (Same logic as before) ...
        ckpt = torch.load(path, map_location=self.device)
        self.vdk.load_state_dict(ckpt['vdk'])
        self.student.load_state_dict(ckpt['student'])
        if 'teacher_policy' in ckpt:
            self.teacher.agent.policy.load_state_dict(ckpt['teacher_policy'])
            self.teacher.agent.critic.load_state_dict(ckpt['teacher_critic'])
            self.teacher.agent.critic_target.load_state_dict(ckpt['teacher_critic_target'])
        else:
             print("[ALL-C] Warning: Teacher checkpoint not found in save file.")
        
    # ------------------------------------------------------------------
    # Training Loop
    # ------------------------------------------------------------------
    def train(self):
        """
        Hybrid Update:
        1. Always: Teacher (SAC) Update (Off-Policy).
        2. Periodic (2048 steps): VDK + Student Update (On-Policy Batch).
        """
        
        # --- 1. Train Teacher (SAC) ---
        # Always runs to keep Q-values updated
        vdk_summary = {}
        
        # --- 2. Check Training Phase ---
        # Phase 1: Pretraining (Wait for start_steps)
        if self.total_steps < self.args.start_steps:
            return vdk_summary
        
        # Phase Transition: Initial Training at start_steps
        # We need a flag or check to run this EXACTLY ONCE when we hit start_steps
        # But since train() is termed "every step", we can check if we just crossed it
        # However, it's safer to check buffer size in Phase 1
        
        is_initial_train = (self.total_steps >= self.args.start_steps and not self.initial_training_done)
        
        # Phase 2: Periodic Training
        is_periodic_train = (len(self.on_policy_buffer) >= self.on_policy_batch_size)
        
        if is_initial_train:
             print(f"[ALL-C] Phase 1 Complete. Running INITIAL TRAINING on {len(self.on_policy_buffer)} samples...")
        elif is_periodic_train:
             print(f"[ALL-C] Phase 2: Running Periodic Update on {len(self.on_policy_buffer)} samples...")
        else:
            return vdk_summary
        # --- Perform VDK/Student Update ---
        
        # 1. Prepare Data for Autoregressive VDK Training
        T_horizon = getattr(self.args, 'horizon', 5)
        
        states_np = np.array([x[0] for x in self.on_policy_buffer])
        actions_np = np.array([x[1] for x in self.on_policy_buffer])
        dones_np = np.array([x[4] for x in self.on_policy_buffer])
        # Rewards/NextStates not strictly needed for VDK autoregression, but for Student
        rewards_np = np.array([x[2] for x in self.on_policy_buffer])
        next_states_np = np.array([x[3] for x in self.on_policy_buffer])
        
        # --- Normalization ---
        # If Initial Training, Compute Mean/Std
        if is_initial_train:
             print("[ALL-C] Computing and Updating State Normalization Stats...")
             full_states = torch.FloatTensor(states_np).to(self.device)
             self.state_mean = full_states.mean(dim=0)
             self.state_std  = full_states.std(dim=0) + 1e-6
             print(f"[ALL-C] State Mean: {self.state_mean[:4]}... | Std: {self.state_std[:4]}...")
        
        # Validate Normalization
        if (self.state_std < 1e-6).any():
             print("[ALL-C] Warning: Low variance features detected. Fixing std to 1.")
             self.state_std[self.state_std < 1e-6] = 1.0
             
        # Identify valid sequence starts
        valid_starts = []
        N = len(self.on_policy_buffer)
        for i in range(N - T_horizon + 1):
             # Check if any done occurs in [i, i+T-2] (intermediate steps cannot be terminal)
             # If done is True at T-1, it's okay (sequence ends).
             segment_dones = dones_np[i : i+T_horizon-1]
             if not np.any(segment_dones):
                  valid_starts.append(i)
        
        if not valid_starts:
             return vdk_summary

        final_vdk_loss = 0
        final_student_loss = 0
        
        # --- A. Train VDK (Dynamics) ---
        # Train for K epochs first
        epochs = self.on_policy_epochs if is_initial_train else self.args.vll_finetune_epochs_dyn
        
        # Use tqdm for progress bar
        pbar_dyn = range(epochs)

        batch_size = len(valid_starts) if is_initial_train else self.on_policy_batch_size
        for epoch in pbar_dyn:
             # Sample batch of sequences
             batch_indices = np.random.choice(valid_starts, size=min(batch_size, len(valid_starts)), replace=True)
             
             s_batch_list = []
             u_batch_list = []
             
             for idx in batch_indices:
                  s_batch_list.append(states_np[idx : idx+T_horizon])
                  u_batch_list.append(actions_np[idx : idx+T_horizon])
                  
             s_batch = torch.FloatTensor(np.array(s_batch_list)).to(self.device)
             u_batch = torch.FloatTensor(np.array(u_batch_list)).to(self.device)
             
             # Normalize Input Batch
             s_batch = (s_batch - self.state_mean) / self.state_std

             self.vdk_optimizer.zero_grad()
             
             # EXACT Pretraining Logic
             loss_total, metrics = compute_autoregressive_loss(
                 self.vdk, s_batch, u_batch, eps=0.5, gamma_loss=0.5, spectral_reg_weight=0.01
             )
             
             loss_total.backward()
             torch.nn.utils.clip_grad_norm_(self.vdk.parameters(), 1.0) # Clip VDK Grads
             self.vdk_optimizer.step()
             
             final_vdk_loss = loss_total.item()
             loss_spec = metrics["spec"]
             
             # Calculate Latent Spread for logging (Diagnostic)
             with torch.no_grad():
                 mu0_re, mu0_im, _, _ = self.vdk.encode(s_batch[:, 0, :])
                 spread_re = mu0_re.var(dim=0).mean().item()
                 spread_im = mu0_im.var(dim=0).mean().item()
                 mean_spread = spread_re + spread_im
                 
             print(f"Epoch {epoch+1}: Loss = {final_vdk_loss:.4f} (Spec: {loss_spec.item():.4f}) | LatentSpread: {mean_spread:.2e}")
             
             if mean_spread < 1e-4:
                 print("WARNING: Latent Space Collapse Detected! (Spread < 1e-4)")
            
        # --- B. Train Student (Value) ---
        # Train for K epochs using the FROZEN updated dynamics
        
        # Restore full next-state tensor for Student update
        ns_t = torch.FloatTensor(next_states_np).to(self.device)
        # Normalize Next States
        ns_t = (ns_t - self.state_mean) / self.state_std
        
        r_t = torch.FloatTensor(rewards_np).to(self.device).unsqueeze(1)
        d_t = torch.FloatTensor(dones_np).to(self.device).unsqueeze(1)
        
        print("\n=== Stage 2: Training Student (Value) ===")
        pbar_student = range(epochs)
        for epoch in pbar_student:
            self.student_optimizer.zero_grad()
            with torch.no_grad():
                 # Encode and Predict Next State using newly trained VDK
                 mu_re, mu_im, _, _ = self.vdk.encode(ns_t)
                 z_next_tilde = torch.cat([mu_re, mu_im], dim=-1)
                 
                 # Solve u*_next with updated dynamics
                 u_next_star = self.solve_analytic_u(mu_re, mu_im)
                 if torch.isnan(u_next_star).any():
                      u_next_star = torch.zeros_like(u_next_star) # Fallback

                 u_next_star = torch.clamp(u_next_star, self.u_min, self.u_max)
                 
                 # Teacher Target Q
                 q1, q2 = self.teacher.agent.critic(ns_t, u_next_star)
                 q_target = torch.min(q1, q2)
                 
                 gamma = self.args.gamma
                 y = r_t + gamma * (1 - d_t) * q_target
                 
                 # Add Control Cost to Target for V
                 u_cost = torch.sum((u_next_star ** 2) * self.R_val, dim=1, keepdim=True)
                 v_target = y + u_cost
            
            # Predict V(z_next) - Student Regression
            # Note: ns_t was already normalized above before the loop
            v_pred = self.student(z_next_tilde.detach()) 
            
            student_loss = F.mse_loss(v_pred, v_target)
            student_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0) # Clip Student Grads
            self.student_optimizer.step()
            final_student_loss = student_loss.item()
            
            print(f"Epoch {epoch+1}: Loss = {final_student_loss:.4f}")
        
        # Clear Buffer
        self.on_policy_buffer = []
        
        if is_initial_train:
             self.initial_training_done = True
        
        vdk_summary = {
            'vdk_loss': final_vdk_loss,
            'student_loss': final_student_loss
        }
    
        return {
            **vdk_summary # Merges vdk_loss, student_loss if present
        }

