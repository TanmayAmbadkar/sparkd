import torch
import numpy as np
import osqp
import scipy.sparse as sparse
from torch.distributions import Normal

class VDK_Runtime:
    """
    Runtime Shield execution for Spectral Koopman Dynamics.
    1. Encodes state s -> mu_t (Re(z)), logvar_t
    2. Calculates Adaptive Safety Margin M_t using spectral variance propagation
    3. Solves QP to find u_safe close to u_rl
    
    Handles the complex-to-real projection implicitly:
    z_real = [Re(z); Im(z)]
    Dynamics: z_{t+1} = Lambda * z_t + B * u
    Safety: w^T z_real + beta >= M_t
    """
    def __init__(self, model, device='cpu', confidence=0.05):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # --- Precompute / Cache Linear Parameters ---
        # Move everything to CPU/Numpy for OSQP
        
        # 1. Dynamics (Spectral)
        # Lambda = mu + i*omega
        # B_c = B_re + i*B_im
        dyn = self.model.dynamics
        
        self.mu_vec    = dyn.mu.detach().cpu().numpy()          # (d,)
        self.omega_vec = dyn.omega.detach().cpu().numpy()       # (d,)
        self.B_re      = dyn.B_re.detach().cpu().numpy()        # (d, m)
        self.B_im      = dyn.B_im.detach().cpu().numpy()        # (d, m)
        
        # 2. CBF Head
        # w is (2d, 1) -> split into w_re (d, 1) and w_im (d, 1)
        w_all = self.model.w.detach().cpu().numpy().flatten()   # (2d,)
        d = self.model.latent_dim
        self.w_re = w_all[:d]  # (d,)
        self.w_im = w_all[d:]  # (d,)
        
        self.beta = self.model.beta.detach().cpu().numpy().item()
        
        # --- QP Constants (Precomputed) ---
        # Constraint: (w^T B_eff) u >= Lower_Bound
        # w^T z_{next} = w_re^T z_{next,re} + w_im^T z_{next,im}
        # z_{next,re} = mu*z_re - omega*z_im + B_re*u
        # z_{next,im} = omega*z_re + mu*z_im + B_im*u
        
        # u-term coefficient vector C (size m):
        # C = w_re^T B_re + w_im^T B_im
        # C_j = sum_k (w_re_k * B_re_kj + w_im_k * B_im_kj)
        self.C_qp = self.w_re @ self.B_re + self.w_im @ self.B_im  # (m,)
        
        # --- Safety Margin Constants ---
        # |Lambda_i|^2 = mu_i^2 + omega_i^2
        self.lambda_sq = self.mu_vec**2 + self.omega_vec**2      # (d,)
        # |w_i|^2      = w_re_i^2 + w_im_i^2
        self.w_sq      = self.w_re**2 + self.w_im**2             # (d,)
        
        # Z-score for confidence
        self.z_score = Normal(0, 1).icdf(torch.tensor(1.0 - confidence)).item()
        
    def solve_shield(self, s_t, u_rl):
        """
        Inputs:
            s_t: np.array (n,) - Observations
            u_rl: np.array (m,) - Proposed Action
            
        Returns:
            u_safe: np.array (m,)
            status: str
        """
        # 1. Encode
        with torch.no_grad():
            s_tensor = torch.tensor(s_t, dtype=torch.float32).unsqueeze(0).to(self.device)
            mu_t, logvar_t = self.model.encode(s_tensor)
            
            # Extract real stats (Im(z_t) is assumed 0)
            z_re_t = mu_t.cpu().numpy().flatten()        # (d,)
            sigma_sq_t = torch.exp(logvar_t).cpu().numpy().flatten() # (d,)
            
            # z_im_t is effectively zero for the current step t
            
        # 2. Adaptive Safety Margin M_t
        # sigma^2_safety = sum_i ( |w_i|^2 * |Lambda_i|^2 * sigma^2_{t,i} )
        # This propagates the variance through one step of dynamics
        safety_var = np.sum(self.w_sq * self.lambda_sq * sigma_sq_t)
        safe_std = np.sqrt(safety_var + 1e-8)
        M_t = self.z_score * safe_std
        
        # 3. Construct QP
        m = len(u_rl)
        P = sparse.csc_matrix(2 * np.eye(m))
        q = -2 * u_rl
        
        # Linear Constraint: C_qp @ u >= Lower_Bound
        # Mean(V(z_{next})) = w^T E[z_{next}] + beta
        # E[z_{next, re}] = mu*z_re - omega*0 + B_re*u = mu*z_re + B_re*u
        # E[z_{next, im}] = omega*z_re + mu*0 + B_im*u = omega*z_re + B_im*u
        
        # Mean(V) = w_re^T(mu*z_re + B_re*u) + w_im^T(omega*z_re + B_im*u) + beta
        #         = (w_re^T B_re + w_im^T B_im) u + (w_re^T(mu*z_re) + w_im^T(omega*z_re)) + beta
        #         = C_qp @ u + Term_State + beta
        
        # We want: Mean(V) >= M_t
        # C_qp @ u >= M_t - Term_State - beta
        
        # Calculate Term_State: sum_i ( w_re_i * mu_i * z_re_i + w_im_i * omega_i * z_re_i )
        # = sum_i z_re_i * ( w_re_i * mu_i + w_im_i * omega_i )
        
        term_state = np.dot(z_re_t, self.w_re * self.mu_vec + self.w_im * self.omega_vec)
        
        lower_bound = M_t - term_state - self.beta
        
        # Clamp for numerical stability
        if np.abs(lower_bound) > 1e6:
            lower_bound = np.clip(lower_bound, -1e6, 1e6)
            
        upper_bound = 1e20
        
        # OSQP Matrices
        C_matrix = self.C_qp.reshape(1, m)
        C_sparse = sparse.csc_matrix(C_matrix)
        
        A_total = sparse.vstack([C_sparse, sparse.eye(m)], format='csc')
        l_total = np.hstack([lower_bound, -np.ones(m)]) # u >= -1
        u_total = np.hstack([upper_bound, np.ones(m)])  # u <= 1
        
        # 4. Solve
        prob = osqp.OSQP()
        try:
            prob.setup(P, q, A_total, l_total, u_total, verbose=False, eps_abs=1e-3, eps_rel=1e-3, check_termination=10)
        except ValueError:
            return np.zeros_like(u_rl), 'infeasible'
            
        res = prob.solve()
        
        if res.info.status == 'solved':
            return res.x, 'feasible'
        else:
            # Relax
            # Try setting M_t = 0 (Neutral safety)
            relaxed_bound = -term_state - self.beta
            l_total[0] = relaxed_bound
            prob.update(l=l_total)
            res = prob.solve()
            
            if res.info.status == 'solved':
                return res.x, 'relaxed'
            else:
                return np.zeros_like(u_rl), 'infeasible'

    def __call__(self, s_t, u_rl):
        return self.solve_shield(s_t, u_rl)
