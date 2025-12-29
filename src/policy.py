import gymnasium as gym
import numpy as np
from typing import Optional, List, Tuple, Union
import scipy
import torch
import time

from pytorch_soft_actor_critic.sac import SAC
from pytorch_soft_actor_critic.replay_memory import ReplayMemory
from ppo import PPO
from koopman.env_model import KoopmanLinearModel
import osqp
import scipy.sparse as sp



class SACPolicy:

    def __init__(self,
                 gym_env: gym.Env,
                 replay_size: int,
                 seed: int,
                 batch_size: int,
                 sac_args):
        self.agent = SAC(gym_env.observation_space.shape[0],
                         gym_env.action_space, sac_args)
        self.memory = ReplayMemory(replay_size, gym_env.observation_space, gym_env.action_space.shape[0], seed)
        self.updates = 0
        self.batch_size = batch_size

    def __call__(self, state: np.ndarray, evaluate: bool = False):
        return self.agent.select_action(state, evaluate = evaluate)

    def add(self, state, action, reward, next_state, done, cost):
        self.memory.push(state, action, reward, next_state, done, cost)

    def train(self):
        ret = self.agent.update_parameters(self.memory, self.batch_size,
                                           self.updates)
        self.updates += 1
        return ret

    def report(self):
        return 0, 0

    def load_checkpoint(self, path):
        self.agent.load_checkpoint(path)


class PPOPolicy:

    def __init__(self,
                 gym_env: gym.Env,
                 replay_size: int,
                 seed: int,
                 batch_size: int,
                 args):
        self.agent = PPO(gym_env.observation_space.shape[0],
                         gym_env.action_space, args)
        self.memory = ReplayMemory(replay_size, gym_env.observation_space, gym_env.action_space.shape[0], seed)
        self.updates = 0
        self.minibatch_size = args.mini_batch_size
        self.batch_size = batch_size

    def __call__(self, state: np.ndarray, evaluate: bool = False):
        return self.agent.select_action(state)[0]

    def add(self, state, action, reward, next_state, done, cost):
        self.memory.push(state, action, reward, next_state, done, cost)

    def train(self):
        ret = self.agent.update_parameters(self.memory, batch_size=self.minibatch_size, epochs = 40)
        self.updates += 1
        return ret

    def report(self):
        return 0, 0

    def load_checkpoint(self, path):
        self.agent.load_checkpoint(path)

# --- Start of updated ProjectionPolicy ---

class ProjectionPolicy:
    def __init__(self,
                 env: KoopmanLinearModel,
                 state_space: gym.Space,
                 action_space: gym.Space,
                 horizon: int,
                 unsafe_polys: List[np.ndarray],
                 safe_polys: List[np.ndarray],
                 transform=lambda x: x):
        self.env = env
        self.horizon = horizon
        self.state_space = state_space
        self.action_space = action_space
        self.unsafe_polys = unsafe_polys
        self.safe_polys = safe_polys
        self.transform = transform
        self.s_dim = self.state_space.shape[0]
        self.u_dim = self.action_space.shape[0]

        # --- Cache Attributes ---
        self._A, self._B, self._c = None, None, None
        
        self._precomputed_F = {}
        self._precomputed_h_base = {} 
        self._precomputed_M = {}
        
        # Cache for nominal error bound (Single V0, eps0)
        self.V0_nominal, self.eps0_nominal = None, None
        self.is_adaptive = False

        # --- Stateful call optimization ---
        self.saved_state = None
        self.saved_action = None
        self.shielded = None

    def update_model(self):
        """
        Pre-computes safety matrices using Single-Step Analytical Propagation.
        """
        print("Pre-computing safety projection matrices...")
        
        # 1. Get Global Dynamics and Single Error Bound (V0, eps0)
        dummy_state = np.zeros(self.s_dim)
        dummy_action = np.zeros(self.u_dim)
        
        # We must retrieve a single (V, eps) pair
        mat_dyn, adaptive_info = self.env.get_matrix_at_point(
            np.concatenate((dummy_state, dummy_action)), self.s_dim
        )
        
        self._A, self._B, self._c = mat_dyn[:, :self.s_dim], mat_dyn[:, self.s_dim:-1], mat_dyn[:, -1]
        
        # adaptive_info is now (V_0, eps_0)
        self.V0_nominal, self.eps0_nominal = adaptive_info
        self.is_adaptive = self.env.adaptive_error
        
        # Logging check
        self._global_eps = self.eps0_nominal
        if self._global_eps.size > 0:
            print(f"Mode: {'ADAPTIVE' if self.is_adaptive else 'FIXED'}. Single Eps (V0, eps0) Retrieved.")
        else:
            print("Error: Empty epsilon vector.")


        s_dim = self.s_dim
        u_dim = self.u_dim

        # Clear old cache
        self._precomputed_F.clear()
        self._precomputed_h_base.clear()
        self._precomputed_M.clear()

        # 3. Pre-compute Propagation (Nominal Path + FIXED Analytical Error)
        V_fixed, eps_fixed = self.V0_nominal, self.eps0_nominal
        
        for poly_idx, poly in enumerate(self.safe_polys):
            P_poly, b_poly = poly[:, :-1], poly[:, -1]

            F = []
            h_base = []
            
            for j in range(1, self.horizon + 1):
                F.append([None] * (j + 1))
                h_base.append([None] * (j + 1))
                
                # Initialization at t=j
                F[j - 1][j] = P_poly
                h_base[j - 1][j] = b_poly
                
                # Backward propagation to t=0
                for t in range(j - 1, -1, -1):
                    # Propagate Dynamics
                    F[j - 1][t] = np.dot(F[j - 1][t + 1], self._A)
                    term_c = np.dot(F[j - 1][t + 1], self._c)
                    
                    if not self.is_adaptive:
                        # FIXED MODE: Add the analytical error buffer NOW (Optimization)
                        # We use the nominal V0, eps0 for pre-computation
                        P_matrix = F[j - 1][t + 1] # P_t+1 in the robust constraint derivation
                        P_rot = np.dot(P_matrix, V_fixed)
                        term_eps = np.dot(np.abs(P_rot), eps_fixed)

                        h_base[j - 1][t] = h_base[j - 1][t + 1] + term_c + term_eps
                    else:
                        # ADAPTIVE MODE: Store nominal path only (Error added in solve)
                        h_base[j - 1][t] = h_base[j - 1][t + 1] + term_c

            self._precomputed_F[poly_idx] = F
            self._precomputed_h_base[poly_idx] = h_base

            # G and M (Action Constraints) remains unchanged
            G = []
            for j in range(1, self.horizon + 1):
                G.append([None] * (j + 1))
                G[j - 1][j] = np.zeros((b_poly.shape[0], u_dim))
                for t in range(j - 1, -1, -1):
                    G[j - 1][t] = np.dot(F[j - 1][t + 1], self._B)
            
            total_vars = self.horizon * u_dim
            n_constraints = self.horizon * P_poly.shape[0]
            M = np.zeros((n_constraints, total_vars))
            
            ind = 0
            step = P_poly.shape[0]
            for j in range(self.horizon):
                G[j] += [np.zeros((P_poly.shape[0], u_dim))] * (self.horizon - j - 1)
                M[ind:ind + step, :] = np.concatenate(G[j][:-1], axis=1)
                ind += step
            
            self._precomputed_M[poly_idx] = M
            
        print("Pre-computation complete.")

    def solve(self, state: np.ndarray,
              action: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool]:
        """
        Solves the safety projection QP with Soft Constraints (Slacks) using 
        Single-Step Adaptive Error Bounds propagated analytically.
        """
        original_state = state.copy()
        shielded = True
        u_dim = self.u_dim
        
        # 0. State Pre-processing
        state = self.transform(state.reshape(1, -1)).reshape(-1,)
        
        if action is None:
            action = np.zeros(u_dim)
            
        if self._A is None:
            raise RuntimeError("Must call .update_model() before .solve().")

        # --- 1. ADAPTIVE LOGIC: Retrieve Single-Step Error Bound ---
        # adaptive_info is a single tuple (V_0, eps_0)
        _, (V_0, eps_0) = self.env.get_matrix_at_point(original_state, self.s_dim)
        
        best_score = np.inf
        best_u0 = None
        SLACK_TOL = 1e-4

        for poly_idx, poly in enumerate(self.safe_polys):
            P_poly, b_poly = poly[:, :-1], poly[:, -1]
            
            # Quick check to skip if hopelessly unsafe
            violation = np.dot(P_poly, state) + b_poly
            if np.any(violation > 0.5): 
                continue
            
            F = self._precomputed_F[poly_idx]
            h_base = self._precomputed_h_base[poly_idx]
            M_safety = self._precomputed_M[poly_idx]

            # --- 2. SETUP QP VARIABLES (Unchanged from base) ---
            n_u_vars = self.horizon * u_dim
            n_slack_vars = self.horizon
            total_vars = n_u_vars + n_slack_vars
            n_safety_rows = M_safety.shape[0]
            faces = P_poly.shape[0]

            # --- CONSTRAINTS (A matrix) ---
            # (A matrix construction logic is omitted for brevity, assumed correct)
            n_action_rows = 2 * n_u_vars
            n_slack_pos_rows = n_slack_vars
            n_total_rows = n_safety_rows + n_action_rows + n_slack_pos_rows
            
            A_osqp = np.zeros((n_total_rows, total_vars))
            A_osqp[:n_safety_rows, :n_u_vars] = M_safety
            
            # Link constraints to slacks
            current_row = 0
            for t in range(self.horizon):
                col_idx = n_u_vars + t
                A_osqp[current_row : current_row + faces, col_idx] = -1.0
                current_row += faces
            
            # Action Limits and Slack Positivity (A matrix)
            A_osqp[n_safety_rows : n_safety_rows + n_u_vars, :n_u_vars] = np.eye(n_u_vars)
            A_osqp[n_safety_rows + n_u_vars : n_safety_rows + 2*n_u_vars, :n_u_vars] = -np.eye(n_u_vars)
            row_start = n_safety_rows + n_action_rows
            A_osqp[row_start:, n_u_vars:] = -np.eye(n_slack_vars)


            # --- 3. CALCULATE ROBUST BIAS (Upper Bound, u_vec) ---
            bias = np.zeros(n_safety_rows)
            ind = 0
            
            for j in range(self.horizon):
                # Nominal Constraint Value: h_base[j][0] already contains the nominal drift/fixed error
                bias_val = h_base[j][0] + np.dot(F[j][0], state)
                
                # Add Adaptive Buffer ONLY if running in Adaptive Mode
                if self.is_adaptive:
                    # Adaptive Mode: Recalculate Error Buffer analytically from single bound (eps_0)
                    
                    adaptive_buffer = np.zeros(faces)
                    for t in range(j):
                        P_matrix = F[j][t+1] 
                        
                        # 1. Rotate the error coefficient into the PCA basis (V_0)
                        P_rot = np.dot(P_matrix, V_0)
                        
                        # 2. Compute the worst-case push from the error at step t
                        step_error = np.dot(np.abs(P_rot), eps_0)
                        
                        # 3. Accumulate the total analytical buffer
                        adaptive_buffer += step_error
                    
                    # Apply the total analytically propagated buffer
                    bias_val += adaptive_buffer
                
                # The constraint is A x <= -bias 
                bias[ind : ind+faces] = bias_val
                ind += faces

            # Construct full Upper Bound vector (u_vec)
            u_vec = np.concatenate([
                -bias,
                np.tile(self.action_space.high, self.horizon),
                -np.tile(self.action_space.low, self.horizon),
                np.zeros(n_slack_vars)
            ])
            l_vec = np.full_like(u_vec, -np.inf)

            # --- 4. OBJECTIVE & SOLVE (Unchanged from base) ---
            n_u_vars = self.horizon * u_dim
            n_slack_vars = self.horizon
            
            P_matrix = np.eye(total_vars) * 1e-6
            P_matrix[:u_dim, :u_dim] = np.eye(u_dim) # Prioritize u_0 tracking
            P_matrix[n_u_vars:, n_u_vars:] = np.eye(n_slack_vars) * 1e10 # High slack penalty
            P_csc = sp.csc_matrix(P_matrix)
            
            q_vec = np.zeros(total_vars)
            q_vec[:u_dim] = -action # Minimize ||u_0 - u_ref||^2
            
            solver = osqp.OSQP()
            solver.setup(P=P_csc, q=q_vec, A=sp.csc_matrix(A_osqp), l=l_vec, u=u_vec, verbose=False)
            res = solver.solve()
            
            # --- 5. RESULT CHECK ---

            if res.info.status == 'solved':
                sol_u = res.x[:n_u_vars]
                sol_slacks = res.x[n_u_vars:]
                immediate_slack = sol_slacks[0]
                
                if np.allclose(sol_slacks[0], 0) <= SLACK_TOL:
                    candidate_u0 = sol_u[:u_dim]
                    candidate_score = np.linalg.norm(candidate_u0 - action)
                    if candidate_score < best_score:
                        best_score = candidate_score
                        best_u0 = candidate_u0

        # --- 6. FINAL DECISION ---
        if best_u0 is None:
            # If no poly returned a zero-slack solution, we fail.
            best_u0 = self.backup(original_state)
            shielded = False 
        
        self.saved_state = original_state
        self.saved_action = best_u0
        self.shielded = shielded
        return best_u0, shielded

    def __call__(self, state: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Main entry point for the shield.
        """
        if self.saved_state is not None and np.allclose(state, self.saved_state):
            return self.saved_action, self.shielded
        return self.solve(state)

    def unsafe(self, state: np.ndarray, action: np.ndarray) -> bool:
        res, shielded = self.solve(state, action=action)
        # If shielded is True, it means we modified the action -> Unsafe nominal
        return not np.allclose(res, action)

    def backup(self, state: np.ndarray, epsilon: float = 0.1) -> np.ndarray:
        """
        Robust Backup Policy ("Run Away"):
        1. Finds the geometric escape direction from the nearest UNSAFE region.
        2. Solves a QP to find a sequence of actions that maximally moves the state
           in that escape direction over the horizon, subject to action bounds.
           
        QP Objective: 
            maximize (projection_of_final_state_on_escape_dir) - penalty * ||u||^2
        """
        # 0. Initial Setup and State Transformation
        with torch.no_grad():
            z = self.transform(state.reshape(1, -1)).reshape(-1,)
        
        s_dim = self.s_dim
        u_dim = self.u_dim
        
        # --- STAGE 1: Find the Escape Direction (from Unsafe Polys) ---
        # Solving geometric QP to find shortest vector TO an unsafe region
        P_stage1 = sp.eye(s_dim, format='csc')
        q_stage1 = np.zeros(s_dim)
        best_val = np.inf
        best_proj = np.zeros(s_dim)
        
        for poly_idx, unsafe_mat in enumerate(self.unsafe_polys):
            unsafe_mat = np.array(unsafe_mat)[:,0,:]
            A_ineq = unsafe_mat[:, :-1]
            # Constraint: A_ineq * (z + d) <= -unsafe_mat[:,-1]
            # A_ineq * d <= -unsafe_mat[:,-1] - A_ineq * z
            b_ineq = -unsafe_mat[:, -1] - (A_ineq @ z)
            
            backup_qp_stage1 = osqp.OSQP()
            backup_qp_stage1.setup(P=P_stage1, q=q_stage1, A=sp.csc_matrix(A_ineq),
                                    l=-np.inf * np.ones_like(b_ineq), u=b_ineq,
                                    verbose=False)
            res = backup_qp_stage1.solve()
            
            if res.info.status == 'solved':
                dist = np.linalg.norm(res.x)
                if dist < best_val:
                    best_val = dist
                    best_proj = res.x
        
        # Escape direction is opposite to the projection vector
        # best_proj points FROM state TO unsafe region.
        # We want to maximize movement in direction -best_proj.
        if np.linalg.norm(best_proj) < 1e-6:
            # Already inside or very close, or no unsafe regions found.
            # If we are effectively "on top" of the unsafe region, picking a random direction
            # or zero might be appropriate. Here we default to zero.
            return np.zeros(self.u_dim)
            
        escape_dir = -best_proj / np.linalg.norm(best_proj)
        
        # --- STAGE 2: Maximize Separation using QP ---
        # Variable: u (sequence of H actions) -> size H * u_dim
        # Dynamics expansion: x_H = A^H x_0 + sum(A^{H-1-i} B u_i) + terms(c)
        # We want to Maximize: escape_dir^T * x_H
        # Equivalent to Minimize: -escape_dir^T * (Linear_Term_of_u) + Regularization
        
        # 2a. Pre-compute the sensitivity of x_H to each u_i
        # cost_vec (linear term q for QP) has size H * u_dim
        
        # We need the 'B' matrices. Since we might be in adaptive mode or not, 
        # we strictly use the *current local linearization* at the state for the backup plan.
        # This is a local approximation.
        dummy_u = np.zeros(u_dim)
        mat_dyn, _ = self.env.get_matrix_at_point(np.concatenate((z, dummy_u)), s_dim)
        A = mat_dyn[:, :s_dim]
        B = mat_dyn[:, s_dim:-1]
        
        q_qp = np.zeros(self.horizon * u_dim)
        
        # x_H term related to u_i is: A^{H-1-i} * B * u_i
        # We want to minimize: -escape_dir^T * (A^{H-1-i} * B) * u_i
        # So q_block_i = - (escape_dir^T * A^{H-1-i} * B)^T
        
        current_A_power = np.eye(s_dim) # Starts as A^0
        
        # We fill from i = H-1 down to 0 (since A power grows as we go back in time relative to u)
        # u_{H-1} has Coeff B (A^0 B)
        # u_{H-2} has Coeff AB (A^1 B)
        for i in range(self.horizon - 1, -1, -1):
            # Term for u_i: A^{H-1-i} * B
            params = (current_A_power @ B)
            
            # Project onto escape direction
            # sensitivity = escape_dir^T @ params  (shape 1 x u_dim)
            sensitivity = escape_dir @ params
            
            # Add to objective (minimize negative projection)
            q_qp[i*u_dim : (i+1)*u_dim] = -sensitivity
            
            # Update A power for next step (going backwards in time)
            current_A_power = current_A_power @ A

        # 2b. Regularization (Smoothness)
        # Min 0.5 * u^T P u
        # We use a small epsilon weight from the method signature/default
        reg_weight = epsilon 
        P_qp = sp.eye(self.horizon * u_dim, format='csc') * reg_weight
        
        # 2c. Constraints (Action Bounds Only)
        # l <= u <= u
        A_constraints = sp.eye(self.horizon * u_dim, format='csc')
        l_bounds = np.tile(self.action_space.low, self.horizon)
        u_bounds = np.tile(self.action_space.high, self.horizon)
        
        # 2d. Solve
        solver = osqp.OSQP()
        solver.setup(P=P_qp, q=q_qp, A=A_constraints, l=l_bounds, u=u_bounds, verbose=False)
        res_qp = solver.solve()
        
        if res_qp.info.status == 'solved':
            # Return only the first action u_0
            return res_qp.x[:u_dim]
        else:
            print("Run Away QP failed. Returning zero action.")
            return np.zeros(u_dim)
        
class CBFPolicy:
    """
    A safety shield using a Control Barrier Function (CBF) with a learned
    Koopman operator.

    This shield ensures safety by solving a small, efficient Quadratic Program (QP)
    at each timestep to find an action that satisfies the CBF condition, keeping
    the system within the safe set.
    """
    def __init__(
        self,
        env: Union[KoopmanLinearModel],
        state_space: gym.Space,
        ori_state_space: gym.Space,
        action_space: gym.Space,
        horizon: int,
        unsafe_polys: List[np.ndarray],
        safe_polys: List[np.ndarray], 
        transform=lambda x: x,
        gamma=0.7
    ):
        """
        Args:
            koopman_model: The trained Koopman model with `transition` and `get_eps` methods.
            state_space: The latent (Koopman) state space.
            action_space: The environment's action space.
            cbf_gamma: A hyperparameter (0 < gamma < 1) that controls how quickly
                       the state is pushed away from the boundary.
            transform: A function to lift the state to the Koopman space.
        """
        

        self.env = env
        self.horizon = horizon
        self.state_space = state_space
        self.ori_state_space = ori_state_space
        self.action_space = action_space
        self.unsafe_polys = unsafe_polys
        self.safe_polys = safe_polys
        self.transform = transform

        self.s_dim = self.state_space.shape[0]
        self.u_dim = self.action_space.shape[0]
        # For caching results
        self.saved_state = None
        self.saved_action = None
        self.shielded = False
        self.gamma = gamma
        
                # --- Placeholders for the pre-computed model and solver ---
        self.precomputed = {}
        self.is_model_updated = False

        # --- For caching results ---
        self.saved_state = None
        self.saved_action = None
        self.shielded = False

    def update_model(self):
        """
        Updates the shield with a new dynamics model. It pre-computes all
        state-independent components for EACH safe polyhedron provided.
        """
        H_max = self.horizon
        
        # Get a single, fixed dynamics model for pre-computation
        z_init = np.zeros(self.s_dim) # Use a zero state for linearization
        a_pi_init = np.zeros(self.u_dim)
        mat_dyn, eps = self.env.get_matrix_at_point(np.concatenate((z_init, a_pi_init)), self.s_dim)
        A, B, c = mat_dyn[:, :self.s_dim], mat_dyn[:, self.s_dim:-1], mat_dyn[:, -1]
        eps_vec = np.full(self.s_dim, float(eps)) if np.isscalar(eps) else np.asarray(eps, float).reshape(-1,)

        # --- Pre-compute for EACH polyhedron ---
        self.solvers = []
        self.precomputed_per_poly = []

        for poly in self.safe_polys:
            # 1. Pre-compute powers and affine terms (same for all polys)
            A_pows = [np.eye(self.s_dim)]
            for _ in range(1, H_max + 1): A_pows.append(A_pows[-1] @ A)
            C_list = [np.zeros(self.s_dim)]
            for j in range(1, H_max + 1): C_list.append(C_list[-1] + A_pows[j - 1] @ c)

            # 2. Pre-compute face info for the current polyhedron
            print(poly.shape)
            P_sel, b_sel = poly[:, :-1].astype(float), poly[:, -1].astype(float)
            faces = [(P_sel[i, :], float(b_sel[i])) for i in range(P_sel.shape[0])]

            def rel_degree(p: np.ndarray) -> Optional[int]:
                M = B.copy()
                for r in range(1, H_max + 1):
                    if np.linalg.norm(p @ M, ord=np.inf) > 1e-4: return r
                    M = A @ M
                return None

            face_info = [(p, b, rel_degree(p)) for (p, b) in faces]
            r_vals = [r for (_, _, r) in face_info if r is not None]
            H_trap_all = max(r_vals) if r_vals else 1

            # 3. Pre-compute tightening terms
            tighten_pref = {tuple(p): np.cumsum([np.abs(p @ A_pows[ell]) @ eps_vec for ell in range(H_max)]) for p, _, _ in face_info}

            # 4. Build global constraint matrices for this polyhedron
            G_rows, M_h_rows, v_h_rows, row_ptrs = [], [], [], [[] for _ in range(H_max)]
            for j in range(1, H_max + 1):
                blocks = [A_pows[j - 1 - t] @ B for t in range(j)]
                phi_j = np.hstack(blocks + [np.zeros((self.s_dim, (H_max - j) * self.u_dim))])
                for p, b, r in face_info:
                    if r is None or j < r: continue
                    G_rows.append(sp.csr_matrix(p @ phi_j))
                    M_h_rows.append(-p @ A_pows[j] + (self.gamma ** j) * p)
                    tighten = tighten_pref[tuple(p)][j - 1]
                    v_h_rows.append(-p @ C_list[j] - b - tighten + (self.gamma ** j) * b)
                    row_ptrs[j - 1].append(len(G_rows) - 1)

            G_all = sp.vstack(G_rows, format="csc") if G_rows else sp.csc_matrix((0, H_max * self.u_dim))
            M_h = np.vstack(M_h_rows) if M_h_rows else np.zeros((0, self.s_dim))
            v_h = np.array(v_h_rows)

            # 5. Setup a dedicated OSQP solver for this polyhedron
            P_blocks = [sp.eye(self.u_dim, format="csc")] + [1e-4 * sp.eye(self.u_dim, format="csc")] * (H_max - 1)
            Pmat = sp.block_diag(P_blocks, format="csc")
            lb_actions = np.tile(self.action_space.low, H_max)
            ub_actions = np.tile(self.action_space.high, H_max)
            A_qp = sp.vstack([G_all, sp.eye(H_max * self.u_dim, format="csc")], format="csc")
            
            solver = osqp.OSQP()
            solver.setup(P=Pmat, q=np.zeros(H_max * self.u_dim), A=A_qp, 
                          l=np.hstack([-np.inf * np.ones(G_all.shape[0]), lb_actions]), 
                          u=np.hstack([np.zeros(G_all.shape[0]), ub_actions]), 
                          verbose=False, polish=False)
            
            self.solvers.append(solver)
            self.precomputed_per_poly.append({
                "M_h": M_h, "v_h": v_h, "row_ptrs": row_ptrs, 
                "H_trap_all": H_trap_all, "ub_actions": ub_actions
            })

        self.is_model_updated = True
        print(f"[RAMPS] Model updated. Pre-computed constraints for {len(self.safe_polys)} polyhedra.")

    def solve(
        self,
        state: np.ndarray,
        action: Optional[np.ndarray] = None,
        debug: bool = False,
    ) -> Tuple[np.ndarray, bool]:
        """
        Solves for a safe action by first selecting the most appropriate safe
        polyhedron and then using its dedicated pre-computed solver.
        """
        if not self.is_model_updated:
            print("Model not initialized. Performing first-time update.")
            self.update_model()

        z = self.transform(state.reshape(1, -1)).reshape(-1,)
        a_pi = np.zeros(self.u_dim, dtype=float) if action is None else np.asarray(action, float)
        # --- 1. Choose the Active Polyhedron based on current state z ---
        inside_candidates, violated = [], []
        for idx, poly in enumerate(self.safe_polys):
            P, b = poly[:, :-1].astype(float), poly[:, -1].astype(float)
            g = P @ z + b
            worst = float(np.max(g))
            if worst <= 1e-9: # Use a small tolerance
                inside_candidates.append((idx, worst))
            else:
                violated.append((idx, worst))

        if inside_candidates:
            # Pick the poly we are IN with the largest interior slack
            chosen_idx = min(inside_candidates, key=lambda t: t[1])[0]
            mode = "inside"
        elif violated:
            # Fallback: pick the most violated poly
            chosen_idx = min(violated, key=lambda t: t[1])[0]
        else:
            # Should not happen if safe_polys is not empty
            print("[RAMPS] WARN: No safe polyhedra found for the current state.")
            return self.backup(state)

        if debug: print(f"[RAMPS] selection mode={mode}, chosen poly idx={chosen_idx}")
        
        # --- 2. Select the correct pre-computed solver and data ---
        solver = self.solvers[chosen_idx]
        data = self.precomputed_per_poly[chosen_idx]

        # --- 3. State-Dependent Calculations ---
        h_all = data['M_h'] @ z + data['v_h']

        # --- 4. Binary Search for the Largest Feasible Horizon ---
        lo, hi = data['H_trap_all'], self.horizon
        bestH, best_u0, best_dev = 0, None, None
        q_new = np.hstack([-a_pi, np.zeros((self.horizon - 1) * self.u_dim)])
        u_base = np.hstack([np.zeros_like(h_all), data['ub_actions']])

        while lo <= hi:
            mid = (lo + hi) // 2
            active_rows = [idx for j in range(mid) for idx in data['row_ptrs'][j]]
            mask = np.full_like(h_all, np.inf)
            if active_rows: mask[active_rows] = h_all[active_rows]
            u_new = u_base.copy()
            u_new[:len(mask)] = mask
            
            solver.update(q=q_new, u=u_new)
            res = solver.solve()

            if res.info.status == "solved":
                bestH, best_u0, best_dev = mid, res.x[:self.u_dim], float(np.linalg.norm(res.x[:self.u_dim] - a_pi))
                lo = mid + 1
            else:
                hi = mid - 1

        # --- 5. Return Result or Fallback ---
        if bestH > 0:
            shielded = best_dev > 1e-8
            if debug: print(f"[RAMPS] Solved. Largest feasible H={bestH}, ||u0-a_pi||={best_dev:.3e}")
            self.saved_state, self.saved_action, self.shielded = state, best_u0, shielded
            return best_u0, shielded

        if debug: print("[RAMPS] No feasible H found. Using backup policy.")
        u0 = self.backup(state)
        self.saved_state, self.saved_action, self.shielded = state, u0, False
        return u0, False
    
    
    def backup(self, state: np.ndarray) -> np.ndarray:
        """
        A robust backup policy that actively steers the system towards safety.
        It finds the most critical safety constraint and chooses an action that
        maximally increases the corresponding barrier function's value.
        """
        z = self.transform(state.reshape(1, -1)).reshape(-1,)
        s_dim = self.state_space.shape[0]
        u_dim = self.action_space.shape[0]

        # 1. Find the most critical safety constraint (the one we are closest to violating)
        min_h_val = np.inf
        most_critical_grad = None
        for poly in self.safe_polys:
            P_poly, b_poly = poly[:, :-1], poly[:, -1]
            for i in range(P_poly.shape[0]):
                p_i, b_i = P_poly[i, :], b_poly[i]
                # h_i(z) = -(p_i^T * z + b_i)
                h_i_z = -(p_i @ z + b_i)
                if h_i_z < min_h_val:
                    min_h_val = h_i_z
                    # The gradient ∇h_i(z) = -p_i points "inward" toward safety
                    most_critical_grad = -p_i

        if most_critical_grad is None:
            # This can happen if the state is somehow outside all defined safe polytopes.
            # Returning a zero action is a reasonable passive fallback.
            return np.zeros(u_dim)

        # 2. Get the B matrix from the Koopman model (linearized around a zero action)
        mat_dyn, _ = self.env.get_matrix_at_point(np.concatenate((z, np.zeros(u_dim))), s_dim)
        B = mat_dyn[:, s_dim:-1]

        # 3. Formulate and solve a QP to find the best recovery action
        # Objective: Find action 'u' that maximizes the rate of safety increase,
        # which is equivalent to maximizing ∇h^T * (Bu).
        # min - (∇h^T * B) * u
        P_backup = sp.csc_matrix((u_dim, u_dim)) # No quadratic term
        q_backup = -(most_critical_grad.T @ B)

        # Constraints are just the action bounds
        A_backup = sp.csc_matrix(sp.eye(u_dim))
        l_backup = self.action_space.low
        u_backup = self.action_space.high
        
        solver = osqp.OSQP()
        solver.setup(P=P_backup, q=q_backup, A=A_backup, l=l_backup, u=u_backup, verbose=False)
        res = solver.solve()

        if res.info.status == 'solved':
            return res.x
        else:
            # If the recovery QP fails (should be rare), return a passive action
            print("WARN: Backup recovery QP failed. Returning zero action.")
            return np.zeros(u_dim)


    def __call__(self, state: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Main entry point for the shield.
        """
        if self.saved_state is not None and np.allclose(state, self.saved_state):
            return self.saved_action, self.shielded
        return self.solve(state)

    def unsafe(self, state: np.ndarray, action: np.ndarray) -> bool:
        """
        Checks if a proposed action is unsafe by seeing if the shield would modify it.
        """
        safe_action, shielded = self.solve(state, action=action)
        return np.linalg.norm(safe_action - action) > 1e-8



class Shield:
    """
    Construct a shield from a neural policy and a safety layer.
    """

    def __init__(
            self,
            shield_policy,
            unsafe_policy = None,
            means: np.ndarray = None, 
            stds: np.ndarray = None):
        self.shield = shield_policy
        self.agent = unsafe_policy
        self.shield_times = 0
        self.backup_times = 0
        self.agent_times = 0
        self.total_time = 0.
        self.means = means
        self.stds = stds

    def __call__(self, state: np.ndarray, action: np.ndarray = None, **kwargs) -> np.ndarray:
        start = time.time()
        if action is not None:
            proposed_action = action
        else:
            proposed_action = self.agent(state, **kwargs)
            
        if self.means is not None:
            state = (state - self.means) / self.stds

        
        if self.shield.unsafe(state, proposed_action):
            act, shielded  = self.shield(state)
            self.shield_times += 1 if shielded else 0
            self.backup_times += 1 if not shielded else 0
            shielded = "SHIELD" if shielded else "BACKUP"
        else:
            act = proposed_action
            shielded = "NEURAL"
            self.agent_times += 1
        end = time.time()
        self.total_time += end - start
        
        # print(f"Shield: {shielded}, Action: {act}, Time: {end - start:.4f}s")
        return act, shielded, np.linalg.norm(act - proposed_action), proposed_action

    def report(self) -> Tuple[int, int]:
        return self.shield_times, self.agent_times, self.backup_times, self.total_time

    def reset_count(self):
        self.shield_times = 0
        self.agent_times = 0
        self.backup_times = 0
        self.total_time = 0

