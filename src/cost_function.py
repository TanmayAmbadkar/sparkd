import gymnasium as gym
import numpy as np
from typing import Optional, List, Tuple
import scipy
import torch
import time

from koopman.env_model import MarsE2cModel
import osqp
import scipy.sparse as sp



class CostFunction:
    def __init__(self,
                 env: MarsE2cModel,
                 state_space: gym.Space,
                 ori_state_space: gym.Space,
                 action_space: gym.Space,
                 horizon: int,
                 unsafe_polys: List[np.ndarray],
                 safe_polys: List[np.ndarray], 
                 transform=lambda x: x):
        self.env = env
        self.horizon = horizon
        self.state_space = state_space
        self.ori_state_space = ori_state_space
        self.action_space = action_space
        self.unsafe_polys = unsafe_polys
        self.safe_polys = safe_polys
        self.transform = transform

    
    def __call__(self, state: np.ndarray,
          action: Optional[np.ndarray] = None,
          debug: bool = False) -> Tuple[np.ndarray, bool]:
        s_dim = self.state_space.shape[0]
        u_dim = self.action_space.shape[0]
        state = self.transform(state.reshape(1, -1)).reshape(-1,)
        if action is None:
            action = np.zeros(u_dim)

        # get linearization at (state, action)
        mat_dyn, eps = self.env.get_matrix_at_point(np.concatenate((state, action)), s_dim)
        A = mat_dyn[:, :s_dim]
        B = mat_dyn[:, s_dim:-1]
        c = mat_dyn[:, -1]

        best_score = 0

        lambda_slack = 1e-4  # slack penalty (tune as needed)
        

        for poly in self.safe_polys:
            P_poly = poly[:, :-1]
            b_poly = poly[:, -1]
            if not np.all(np.dot(P_poly, state) + b_poly <= 0.0):
                print("State is not in the safe polytope, skipping...")
                continue

            # === Build safety constraints over the horizon ===
            F = []
            G = []
            h = []
            for j in range(1, self.horizon + 1):
                F.append([None] * (j + 1))
                G.append([None] * (j + 1))
                h.append([None] * (j + 1))
                F[j-1][j] = P_poly
                G[j-1][j] = np.zeros((b_poly.shape[0], u_dim))
                h[j-1][j] = b_poly
                for t in range(j - 1, -1, -1):
                    F[j-1][t] = np.dot(F[j-1][t+1], A)
                    G[j-1][t] = np.dot(F[j-1][t+1], B)
                    epsmax = np.dot(np.abs(F[j-1][t+1]), eps)
                    h[j-1][t] = np.dot(F[j-1][t+1], c) + h[j-1][t+1] + epsmax

            n_con = self.horizon * P_poly.shape[0] + 2 * self.horizon * u_dim  # total constraints
            total_vars = self.horizon * u_dim  # action variables

            # === Assemble constraint matrices ===
            M = np.zeros((n_con, total_vars))
            bias = np.zeros(n_con)
            ind = 0
            step = P_poly.shape[0]
            for j in range(self.horizon):
                G[j] += [np.zeros((P_poly.shape[0], u_dim))] * (self.horizon - j - 1)
                M[ind:ind+step, :] = np.concatenate(G[j][:-1], axis=1)
                bias[ind:ind+step] = h[j][0] + np.dot(F[j][0], state)
                ind += step

            # Add action bound constraints.
            ind2 = 0
            for j in range(self.horizon):
                M[ind:ind+u_dim, ind2:ind2+u_dim] = np.eye(u_dim)
                bias[ind:ind+u_dim] = -self.action_space.high
                ind += u_dim
                M[ind:ind+u_dim, ind2:ind2+u_dim] = -np.eye(u_dim)
                bias[ind:ind+u_dim] = self.action_space.low
                ind += u_dim
                ind2 += u_dim

            # ----------- SLACK QP LOGIC BELOW --------------------
            # Add slack variable s >= 0 for each constraint
            slack_size = n_con

            # ========== FIXED QP (u1,...u_{H-1}, s) ==========
            fixed_total = (self.horizon - 1) * u_dim
            M_first = M[:, :u_dim]
            M_rest  = M[:, u_dim:]
            new_bias = bias + M_first @ action

            # QP variables: [u_rest; slack]
            n_fixed = fixed_total
            n_var_fixed = n_fixed + slack_size

            # Objective: tiny penalty on action, larger penalty on slack
            P_fixed = np.eye(n_var_fixed) * 1e-6
            q_fixed = np.zeros(n_var_fixed)
            q_fixed[n_fixed:] = lambda_slack

            # Constraints: G_fixed @ u_rest - s <= h_fixed, s >= 0
            # [M_rest | -I] @ [u_rest; slack] <= h_fixed
            G_fixed_qp = sp.hstack([sp.csc_matrix(M_rest), -sp.eye(slack_size)], format='csc')
            l_fixed = -np.inf * np.ones(slack_size)
            u_fixed = -new_bias

            # Slack bounds: s >= 0
            G_slack_identity = sp.hstack([sp.csc_matrix((slack_size, n_fixed)), sp.eye(slack_size)], format='csc')
            l_slack = np.zeros(slack_size)
            u_slack = np.inf * np.ones(slack_size)

            # Stack all constraints
            A_fixed_total = sp.vstack([G_fixed_qp, G_slack_identity], format='csc')
            l_fixed_total = np.hstack([l_fixed, l_slack])
            u_fixed_total = np.hstack([u_fixed, u_slack])

            fixed_solver = osqp.OSQP()
            fixed_solver.setup(P=sp.csc_matrix(P_fixed),
                            q=q_fixed,
                            A=A_fixed_total,
                            l=l_fixed_total,
                            u=u_fixed_total,
                            warm_start=False,
                            verbose=False)
            res_fixed = fixed_solver.solve()
            slacks = res_fixed.x[-slack_size:]
            
            if sum(slacks) >= best_score:
                
                # No slack needed, so we can use the action directly
                best_score = sum(slacks) if sum(slacks) > 0 else 0.0


        if best_score == np.inf:
            best_score = 0.0
        
        return best_score

