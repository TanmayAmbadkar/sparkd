import gymnasium as gym
import numpy as np
from typing import Optional, List, Tuple
import scipy
import torch
import time

from koopman.env_model import KoopmanLinearModel  # Assuming this is your model definition
import osqp
import scipy.sparse as sp
class CostFunctionWP:
    def __init__(self,
                 env: KoopmanLinearModel,
                 state_space: gym.Space,
                 ori_state_space: gym.Space,
                 action_space: gym.Space,
                 horizon: int,
                 unsafe_polys: List[np.ndarray],
                 safe_polys: List[np.ndarray],
                 transform=lambda x: x,
                 mean: np.ndarray = None,
                 std: np.ndarray = None):
        self.env = env
        self.horizon = horizon
        self.state_space = state_space
        self.ori_state_space = ori_state_space
        self.action_space = action_space
        self.unsafe_polys = unsafe_polys
        self.safe_polys = safe_polys
        self.transform = transform
        self.mean = mean
        self.std = std

        # Storage for pre-computed constraint matrices
        self.precomputed_FGH = []
        
        # Perform the initial pre-computation
        self.update_model_constraints()

    def update_model_constraints(self):
        """
        Pre-computes the F, G, and h matrices for dynamics propagation.
        This should be called whenever the underlying dynamics model (A, B, c) changes.
        """
        print("Updating model constraints (pre-computing F, G, h)...")
        s_dim = self.state_space.shape[0]
        u_dim = self.action_space.shape[0]

        # Get the global, state-independent model matrices
        mat_dyn, eps = self.env.get_matrix_at_point(np.zeros(s_dim + u_dim), s_dim)
        A = mat_dyn[:, :s_dim]
        B = mat_dyn[:, s_dim:-1]
        c = mat_dyn[:, -1]

        self.precomputed_FGH.clear()

        for poly in self.safe_polys:
            P_poly = poly[:, :-1]
            b_poly = poly[:, -1]

            # === Build and store F, G, h matrices over the horizon ===
            F, G, h = [], [], []
            for j in range(1, self.horizon + 1):
                F.append([None] * (j + 1))
                G.append([None] * (j + 1))
                h.append([None] * (j + 1))
                F[j - 1][j] = P_poly
                G[j - 1][j] = np.zeros((b_poly.shape[0], u_dim))
                h[j - 1][j] = b_poly
                for t in range(j - 1, -1, -1):
                    F[j - 1][t] = np.dot(F[j - 1][t + 1], A)
                    G[j - 1][t] = np.dot(F[j - 1][t + 1], B)
                    epsmax = np.dot(np.abs(F[j - 1][t + 1]), eps)
                    h[j - 1][t] = np.dot(F[j - 1][t + 1], c) + h[j - 1][t + 1] + epsmax
            
            self.precomputed_FGH.append({
                'P_poly': P_poly,
                'b_poly': b_poly,
                'F': F,
                'G': G,
                'h': h
            })
        print("Finished updating model constraints.")

    def __call__(self, state: np.ndarray,
                    action: Optional[np.ndarray] = None,
                    debug: bool = False) -> float:
        s_dim = self.state_space.shape[0]
        u_dim = self.action_space.shape[0]

        # Normalize state
        processed_state = (state - self.mean) / (self.std + 1e-8)
        processed_state = self.transform(processed_state.reshape(1, -1)).reshape(-1,)
        if action is None:
            action = np.zeros(u_dim)

        # NEW: Track minimum violation across all polytopes
        min_violation = float('inf')
        lambda_slack = 1e-4

        # Loop through the pre-computed F, G, h for each safe polytope
        for precomp in self.precomputed_FGH:
            P_poly = precomp['P_poly']
            b_poly = precomp['b_poly']
            F, G, h = precomp['F'], precomp['G'], precomp['h']

            # Check if the current state is inside the polytope
            if not np.all(np.dot(P_poly, processed_state) + b_poly <= 0.0):
                if debug:
                    print("State is not in the safe polytope, skipping...")
                continue

            # === Assemble full constraint matrices M and bias on the fly ===
            n_safety_con = self.horizon * P_poly.shape[0]
            n_action_con = 2 * self.horizon * u_dim
            n_con = n_safety_con + n_action_con
            total_vars = self.horizon * u_dim

            M = np.zeros((n_con, total_vars))
            bias = np.zeros(n_con)
            
            # Assemble safety constraints
            ind = 0
            step = P_poly.shape[0]
            for j in range(self.horizon):
                G_j_padded = G[j][:-1] + [np.zeros((P_poly.shape[0], u_dim))] * (self.horizon - j -1)
                M[ind:ind + step, :] = np.concatenate(G_j_padded, axis=1)
                bias[ind:ind + step] = h[j][0] + np.dot(F[j][0], processed_state)
                ind += step
            
            # Assemble action bound constraints
            ind2 = 0
            for j in range(self.horizon):
                M[ind:ind + u_dim, ind2:ind2 + u_dim] = np.eye(u_dim)
                bias[ind:ind + u_dim] = -self.action_space.high
                ind += u_dim
                M[ind:ind + u_dim, ind2:ind2 + u_dim] = -np.eye(u_dim)
                bias[ind:ind + u_dim] = self.action_space.low
                ind += u_dim
                ind2 += u_dim

            # ----------- SLACK QP LOGIC BELOW --------------------
            slack_size = n_con
            M_first = M[:, :u_dim]
            M_rest = M[:, u_dim:]
            new_bias = bias + M_first @ action

            n_fixed = (self.horizon - 1) * u_dim
            n_var_fixed = n_fixed + slack_size
            
            P_fixed = sp.eye(n_var_fixed) * 1e-6
            q_fixed = np.zeros(n_var_fixed)
            q_fixed[n_fixed:] = lambda_slack

            G_fixed_qp = sp.hstack([sp.csc_matrix(M_rest), -sp.eye(slack_size)], format='csc')
            l_fixed = -np.inf * np.ones(slack_size)
            u_fixed = -new_bias

            G_slack_identity = sp.hstack([sp.csc_matrix((slack_size, n_fixed)), sp.eye(slack_size)], format='csc')
            l_slack = np.zeros(slack_size)
            u_slack = np.inf * np.ones(slack_size)

            A_fixed_total = sp.vstack([G_fixed_qp, G_slack_identity], format='csc')
            l_fixed_total = np.hstack([l_fixed, l_slack])
            u_fixed_total = np.hstack([u_fixed, u_slack])

            fixed_solver = osqp.OSQP()
            fixed_solver.setup(P=sp.csc_matrix(P_fixed), q=q_fixed, A=A_fixed_total, l=l_fixed_total, u=u_fixed_total, warm_start=False, verbose=False)
            res_fixed = fixed_solver.solve()
            
            if res_fixed.info.status_val != 1: # Check for solver failure
                continue

            slacks = res_fixed.x[-slack_size:]
            current_violation = np.mean(slacks)
            
            # Early return if perfect safety found in ANY polytope
            if current_violation <= 1e-6:  # Effectively zero (accounting for numerical precision)
                if debug:
                    print(f"Found polytope with zero violations! Returning 0.")
                return 0.0
            
            # Track minimum violation across polytopes
            min_violation = min(min_violation, current_violation)

        # Return minimum violation found, or 0 if no polytopes were feasible
        return min_violation if min_violation != float('inf') else 0.0
    
    
class CostFunctionCBF:
    """
    A cost function that measures safety violations using Control Barrier Function (CBF) constraints.
    Fixes the first action and measures the minimum slack needed for future actions to satisfy CBF constraints.
    """
    
    def __init__(
        self,
        env,  # KoopmanLinearModel or FixedLinearModel
        state_space: gym.Space,
        ori_state_space: gym.Space,
        action_space: gym.Space,
        horizon: int,
        unsafe_polys: List[np.ndarray],
        safe_polys: List[np.ndarray],
        transform=lambda x: x,
        mean: np.ndarray = None,
        std: np.ndarray = None,
        gamma: float = 0.7,
    ):
        """
        Args:
            env: The Koopman/linear dynamics model
            state_space: The latent (Koopman) state space
            ori_state_space: The original state space
            action_space: The environment's action space
            horizon: Planning horizon (total timesteps including first action)
            unsafe_polys: List of unsafe polytopes (not used in this implementation)
            safe_polys: List of safe polytopes defined as [P, b] where Px + b <= 0
            transform: Function to lift state to Koopman space
            mean: Normalization mean (optional)
            std: Normalization std (optional)
            gamma: CBF decay rate (0 < gamma < 1)
        """
        self.env = env
        self.horizon = horizon
        self.state_space = state_space
        self.ori_state_space = ori_state_space
        self.action_space = action_space
        self.unsafe_polys = unsafe_polys
        self.safe_polys = safe_polys
        self.transform = transform
        self.mean = mean
        self.std = std
        self.gamma = gamma
        
        self.s_dim = self.state_space.shape[0]
        self.u_dim = self.action_space.shape[0]
        
        # Storage for pre-computed constraint matrices (per polytope)
        self.precomputed_per_poly = []
        self.is_model_updated = False
        
        # Perform initial pre-computation
        self.update_model_constraints()
    
    def update_model_constraints(self):
        """
        Pre-computes CBF constraint matrices for all safe polytopes.
        Similar to CBFPolicy.update_model() but adapted for cost evaluation.
        """
        print("Updating CBF model constraints...")
        H_max = self.horizon
        
        # Get fixed dynamics model
        z_init = np.zeros(self.s_dim)
        a_init = np.zeros(self.u_dim)
        mat_dyn, eps = self.env.get_matrix_at_point(
            np.concatenate((z_init, a_init)), self.s_dim
        )
        A = mat_dyn[:, :self.s_dim]
        B = mat_dyn[:, self.s_dim:-1]
        c = mat_dyn[:, -1]
        eps_vec = np.full(self.s_dim, float(eps)) if np.isscalar(eps) else np.asarray(eps, float).reshape(-1,)
        
        self.precomputed_per_poly.clear()
        
        # Pre-compute for each safe polytope
        for poly_idx, poly in enumerate(self.safe_polys):
            # 1. Pre-compute matrix powers and affine terms
            A_pows = [np.eye(self.s_dim)]
            for _ in range(1, H_max + 1):
                A_pows.append(A_pows[-1] @ A)
            
            C_list = [np.zeros(self.s_dim)]
            for j in range(1, H_max + 1):
                C_list.append(C_list[-1] + A_pows[j - 1] @ c)
            
            # 2. Extract polytope faces
            P_sel = poly[:, :-1].astype(float)
            b_sel = poly[:, -1].astype(float)
            faces = [(P_sel[i, :], float(b_sel[i])) for i in range(P_sel.shape[0])]
            
            # 3. Compute relative degree for each face
            def rel_degree(p: np.ndarray) -> Optional[int]:
                M = B.copy()
                for r in range(1, H_max + 1):
                    if np.linalg.norm(p @ M, ord=np.inf) > 1e-4:
                        return r
                    M = A @ M
                return None
            
            face_info = [(p, b, rel_degree(p)) for (p, b) in faces]
            r_vals = [r for (_, _, r) in face_info if r is not None]
            H_trap_all = max(r_vals) if r_vals else 1
            
            # 4. Pre-compute tightening terms
            tighten_pref = {
                tuple(p): np.cumsum([
                    np.abs(p @ A_pows[ell]) @ eps_vec 
                    for ell in range(H_max)
                ])
                for p, _, _ in face_info
            }
            
            # 5. Build constraint matrices for ALL timesteps up to horizon
            # We'll split them into first action and future actions
            G_first_rows = []  # Influence of first action u_0
            G_future_rows = []  # Influence of future actions u_1, ..., u_{H-1}
            M_h_rows = []
            v_h_rows = []
            
            for j in range(1, H_max + 1):
                # Build phi_j: the control influence matrix at timestep j
                # phi_j has blocks for [u_0, u_1, ..., u_{j-1}]
                blocks = [A_pows[j - 1 - t] @ B for t in range(j)]
                
                for p, b, r in face_info:
                    if r is None or j < r:
                        continue
                    
                    # Extract influence of first action (u_0)
                    if j >= 1:
                        G_first = p @ blocks[0]
                        G_first_rows.append(G_first)
                    else:
                        G_first_rows.append(np.zeros(self.u_dim))
                    
                    # Extract influence of future actions (u_1, ..., u_{j-1})
                    if j > 1:
                        future_blocks = blocks[1:] + [np.zeros((self.s_dim, self.u_dim))] * (H_max - j)
                        G_future = np.hstack([p @ block for block in future_blocks])
                    else:
                        G_future = np.zeros((H_max - 1) * self.u_dim)
                    G_future_rows.append(sp.csr_matrix(G_future))
                    
                    # State-dependent terms
                    M_h_rows.append(-p @ A_pows[j] + (self.gamma ** j) * p)
                    
                    tighten = tighten_pref[tuple(p)][j - 1]
                    v_h_rows.append(-p @ C_list[j] - b - tighten + (self.gamma ** j) * b)
            
            G_first = np.vstack(G_first_rows) if G_first_rows else np.zeros((0, self.u_dim))
            G_future = sp.vstack(G_future_rows, format="csc") if G_future_rows else sp.csc_matrix((0, (H_max - 1) * self.u_dim))
            M_h = np.vstack(M_h_rows) if M_h_rows else np.zeros((0, self.s_dim))
            v_h = np.array(v_h_rows)
            
            self.precomputed_per_poly.append({
                "P_poly": P_sel,
                "b_poly": b_sel,
                "G_first": G_first,      # Shape: (n_constraints, u_dim)
                "G_future": G_future,    # Shape: (n_constraints, (H-1)*u_dim)
                "M_h": M_h,              # Shape: (n_constraints, s_dim)
                "v_h": v_h,              # Shape: (n_constraints,)
            })
        
        self.is_model_updated = True
        print(f"Finished updating CBF constraints for {len(self.safe_polys)} polytopes.")
    
    def __call__(
        self,
        state: np.ndarray,
        action: Optional[np.ndarray] = None,
        debug: bool = False
    ) -> float:
        """
        Computes the cost (minimum slack violation) for a given state-action pair.
        The first action is FIXED, and we solve for future actions that minimize violations.
        
        Args:
            state: Current state
            action: Proposed first action (if None, uses zero action)
            debug: Print debug information
            
        Returns:
            Minimum average slack across all polytopes (0 if safe, >0 if violations exist)
        """
        if not self.is_model_updated:
            self.update_model_constraints()
        
        # Normalize and transform state
        if self.mean is not None and self.std is not None:
            processed_state = (state - self.mean) / (self.std + 1e-8)
        else:
            processed_state = state
        
        z = self.transform(processed_state.reshape(1, -1)).reshape(-1,)
        
        if action is None:
            action = np.zeros(self.u_dim)
        
        min_violation = float('inf')
        
        # Evaluate cost for each safe polytope
        for poly_idx, precomp in enumerate(self.precomputed_per_poly):
            P_poly = precomp['P_poly']
            b_poly = precomp['b_poly']
            
            # Check if current state is inside polytope
            g = P_poly @ z + b_poly
            if not np.all(g <= 1e-9):
                if debug:
                    print(f"Polytope {poly_idx}: State not inside, skipping.")
                continue
            
            G_first = precomp['G_first']
            G_future = precomp['G_future']
            M_h = precomp['M_h']
            v_h = precomp['v_h']
            
            n_constraints = G_first.shape[0]
            if n_constraints == 0:
                if debug:
                    print(f"Polytope {poly_idx}: No CBF constraints.")
                continue
            
            # Compute state-dependent constraint bounds
            h_all = M_h @ z + v_h
            
            # Update bounds with fixed first action: h_all - G_first @ action
            new_bias = h_all - G_first @ action
            
            # Build slack QP for future actions only
            # Variables: [u_1, ..., u_{H-1}, slack_1, ..., slack_n]
            n_future_actions = (self.horizon - 1) * self.u_dim
            n_slack = n_constraints
            total_vars = n_future_actions + n_slack
            
            # === MODIFICATION HERE ===
            # Objective: Minimize quadratic slack penalty and regularize actions
            
            # Define penalties
            action_regularization = 1e-6  # "very small norm of actions"
            slack_penalty = 1.0           # Slack penalty set to 1
            
            P_diag = np.zeros(total_vars)
            
            # Add small quadratic cost to future actions for regularization
            # This corresponds to the (u_1, ..., u_{H-1}) variables
            P_diag[:n_future_actions] = action_regularization
            
            # Add quadratic cost to slacks
            # This corresponds to the (slack_1, ..., slack_n) variables
            P_diag[-n_slack:] = slack_penalty
            
            # Create the sparse quadratic cost matrix P
            P = sp.diags(P_diag, format='csc')
            
            # The linear cost q is now all zeros
            # The objective is 0.5 * x.T @ P @ x
            q = np.zeros(total_vars)
            # === END OF MODIFICATION ===
            
            # Constraints: G_future @ u_future - slack <= new_bias
            A_cbf = sp.hstack([G_future, -sp.eye(n_slack)], format='csc')
            l_cbf = -np.inf * np.ones(n_constraints)
            u_cbf = new_bias
            
            # Action bounds for future actions
            A_action = sp.hstack([
                sp.eye(n_future_actions),
                sp.csc_matrix((n_future_actions, n_slack))
            ], format='csc')
            l_action = np.tile(self.action_space.low, self.horizon - 1)
            u_action = np.tile(self.action_space.high, self.horizon - 1)
            
            # Slack non-negativity: slack >= 0
            A_slack = sp.hstack([
                sp.csc_matrix((n_slack, n_future_actions)),
                sp.eye(n_slack)
            ], format='csc')
            l_slack = np.zeros(n_slack)
            u_slack = np.inf * np.ones(n_slack)
            
            # Combine all constraints
            A_total = sp.vstack([A_cbf, A_action, A_slack], format='csc')
            l_total = np.hstack([l_cbf, l_action, l_slack])
            u_total = np.hstack([u_cbf, u_action, u_slack])
            
            # Solve QP
            solver = osqp.OSQP()
            solver.setup(
                P=P, q=q, A=A_total, l=l_total, u=u_total,
                warm_start=False, verbose=False, polish=False
            )
            res = solver.solve()
            
            if res.info.status_val != 1:  # Not solved
                if debug:
                    print(f"Polytope {poly_idx}: Solver failed with status {res.info.status}")
                continue
            
            # Extract slack values
            slacks = res.x[-n_slack:]
            # Since the objective is now quadratic (0.5 * s^2), the average
            # slack is still the best measure of "violation".
            current_violation = np.mean(slacks)
            
            if debug:
                print(f"Polytope {poly_idx}: Average slack = {current_violation:.6f}")
            
            # Early return if perfect safety found
            if current_violation <= 1e-6:
                if debug:
                    print(f"Polytope {poly_idx}: Zero violations! Returning 0.")
                return 0.0
            
            min_violation = min(min_violation, current_violation)
        
        # Return minimum violation or 0 if no polytopes were feasible
        return min_violation if min_violation != float('inf') and min_violation > 1e-6 else 0.0