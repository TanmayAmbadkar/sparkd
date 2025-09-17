import gymnasium as gym
import numpy as np
from typing import Optional, List, Tuple
import scipy
import torch
import time

from koopman.env_model import KoopmanLinearModel  # Assuming this is your model definition
import osqp
import scipy.sparse as sp

class CostFunction:
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