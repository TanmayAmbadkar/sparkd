import gymnasium as gym
import numpy as np
from typing import Optional, List, Tuple
import scipy
import torch
import time

from pytorch_soft_actor_critic.sac import SAC
from pytorch_soft_actor_critic.replay_memory import ReplayMemory
from ppo import PPO
from koopman.env_model import MarsE2cModel
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
        ret = self.agent.update_parameters(self.memory, batch_size=self.minibatch_size, epochs = 10)
        self.updates += 1
        return ret

    def report(self):
        return 0, 0

    def load_checkpoint(self, path):
        self.agent.load_checkpoint(path)


class ProjectionPolicy:
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

    
    def backup(self, state: np.ndarray, epsilon: float = 0.1) -> np.ndarray:
        """
        Chooses a backup action by finding a smooth control sequence that pushes
        the system away from the nearest unsafe polygon.

        This is a two-stage process:
        1. A QP finds the geometric "escape vector" from the current state.
        2. A second QP finds a smooth, full-horizon action sequence that
        aligns with this escape vector, avoiding "bang-bang" control.

        Args:
            state: The current original system state.
            epsilon: Regularization weight. Higher values lead to smoother,
                    smaller-norm actions.

        Returns:
            The first action (u_0) of the optimal safe sequence.
        """
        # --- Stage 1: Find Geometric Escape Vector (Identical to your original code) ---
        with torch.no_grad():
            z = self.transform(state.reshape(1, -1)).reshape(-1,)
        
        s_dim = self.state_space.shape[0]
        P_stage1 = sp.eye(s_dim, format='csc')
        q_stage1 = np.zeros(s_dim)
        best_val = np.inf
        best_proj = np.zeros(s_dim)

        for unsafe_mat in self.unsafe_polys:
            A_ineq = unsafe_mat[:, :-1]
            b_ineq = -unsafe_mat[:, -1] - (A_ineq @ z)
            
            # This setup is inefficient; ideally the solver is initialized once.
            # But keeping it for consistency with your original code.
            backup_qp_stage1 = osqp.OSQP()
            backup_qp_stage1.setup(P=P_stage1, q=q_stage1, A=sp.csc_matrix(A_ineq),
                                l=-np.inf * np.ones_like(b_ineq), u=b_ineq,
                                verbose=False)
            res = backup_qp_stage1.solve()
            
            if res.info.status == 'solved' and np.linalg.norm(res.x) < best_val:
                best_val = np.linalg.norm(res.x)
                best_proj = res.x

        if np.linalg.norm(best_proj) < 1e-6:
            # Could not find a valid escape direction
            return np.zeros(self.action_space.shape[0])
            
        best_proj /= np.linalg.norm(best_proj)

        # --- Stage 2: Solve for a Smooth Action Sequence (QP instead of LP) ---
        u_dim = self.action_space.shape[0]
        total_control_dim = self.horizon * u_dim

        # Get linearization and compute the linear part of the cost vector `m`
        # (Identical to your original code)
        point = np.concatenate((z, np.zeros(u_dim)))
        mat, _ = self.env.get_matrix_at_point(point, s_dim)
        A_lin = mat[:, :s_dim]
        B_lin = mat[:, s_dim:-1]
        
        m = np.zeros(total_control_dim)
        for i in range(self.horizon):
            A_pow = np.linalg.matrix_power(A_lin, self.horizon - i - 1)
            m[i*u_dim:(i+1)*u_dim] = (B_lin.T @ A_pow.T @ (-best_proj)).T

        # --- QP Formulation ---
        # Objective: min -m^T * U + epsilon * ||U||^2
        # Standard form: min 0.5 * U^T * P * U + q^T * U
        
        # Quadratic part: P = 2 * epsilon * I
        P_stage2 = sp.csc_matrix(2 * epsilon * sp.eye(total_control_dim))
        
        # Linear part: q = -m
        q_stage2 = -m
        
        # Constraints are just the action bounds
        A_stage2 = sp.csc_matrix(sp.eye(total_control_dim))
        l_stage2 = np.tile(self.action_space.low, self.horizon)
        u_stage2 = np.tile(self.action_space.high, self.horizon)
        
        # Setup and solve the QP
        backup_qp_stage2 = osqp.OSQP()
        backup_qp_stage2.setup(P=P_stage2, q=q_stage2, A=A_stage2, 
                            l=l_stage2, u=u_stage2, verbose=False)
        
        res_final = backup_qp_stage2.solve()
        
        if res_final.info.status == 'solved':
            full_action_sequence = res_final.x
            return full_action_sequence[:u_dim]
        else:
            print("WARN: Backup QP failed to find a smooth action. Returning zero action.")
            return np.zeros(u_dim)
    

    def solve(self, state: np.ndarray,
          action: Optional[np.ndarray] = None,
          debug: bool = False) -> Tuple[np.ndarray, bool]:
        original_state = state.copy()
        shielded = True
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

        best_score = np.inf
        best_u0 = None

        for poly in self.safe_polys:
            P_poly = poly[:, :-1]
            b_poly = poly[:, -1]
            
            # Skip if the current state is not in the safe polytope.
            if not np.all(np.dot(P_poly, state) + b_poly <= 0.0):
                continue
                
            # === Build safety constraints over the horizon ===
            # ... (This part of the code remains identical) ...
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
            

            # === Assemble the full QP constraints ===
            # ... (This part of the code also remains identical) ...
            n_constraints = self.horizon * P_poly.shape[0] + 2 * self.horizon * u_dim
            total_vars = self.horizon * u_dim
            M = np.zeros((n_constraints, total_vars))
            bias = np.zeros(n_constraints)
            ind = 0
            step = P_poly.shape[0]
            for j in range(self.horizon):
                G[j] += [np.zeros((P_poly.shape[0], u_dim))] * (self.horizon - j - 1)
                M[ind:ind+step, :] = np.concatenate(G[j][:-1], axis=1)
                bias[ind:ind+step] = h[j][0] + np.dot(F[j][0], state)
                ind += step
            ind2 = 0
            for j in range(self.horizon):
                M[ind:ind+u_dim, ind2:ind2+u_dim] = np.eye(u_dim)
                bias[ind:ind+u_dim] = -self.action_space.high
                ind += u_dim
                M[ind:ind+u_dim, ind2:ind2+u_dim] = -np.eye(u_dim)
                bias[ind:ind+u_dim] = self.action_space.low
                ind += u_dim
                ind2 += u_dim

            ### START OF REPLACEMENT ###
            # --- LP Feasibility Check: u1..u_{H-1} ---
            # We replace the Fixed-QP with a more efficient LP feasibility check.
            fixed_total = (self.horizon - 1) * u_dim
            M_first = M[:, :u_dim]
            M_rest = M[:, u_dim:]
            # The constraints are M_rest @ u_rest + (bias + M_first @ u0) <= 0
            # which is M_rest @ u_rest <= - (bias + M_first @ u0)
            new_bias = bias + M_first @ action

            # For a feasibility LP, the cost vector 'c' is all zeros.
            c_lp = np.zeros(fixed_total)
            
            # The constraints are A_ub @ x <= b_ub
            A_ub_lp = M_rest
            b_ub_lp = -new_bias

            # Solve the LP. We only care if it terminated successfully, which
            # indicates that a feasible solution was found.
            # The 'highs' method is the current default and is very efficient.
            res_lp = scipy.optimize.linprog(c=c_lp, A_ub=A_ub_lp, b_ub=b_ub_lp, method='highs', bounds = (self.action_space.low[0], self.action_space.high[0]))

            # The 'success' attribute is True if a feasible solution was found.
            fixed_feasible = res_lp.success
            ### END OF REPLACEMENT ###

            if fixed_feasible:
                candidate_u0 = action.copy()
                candidate_score = 0.0
                shielded = False
            else:
                # --- Full-QP: optimize [u0; u1..] ---
                # This part remains the same, as it's a true QP.
                total_vars = self.horizon * u_dim
                P_full = 1e-6 * np.eye(total_vars)
                P_full[:u_dim, :u_dim] = np.eye(u_dim)  # Hessian for u0
                q_full = np.zeros((self.horizon)*u_dim)
                q_full[:u_dim] = -action
                A_full = sp.csc_matrix(M)
                l_full = -np.inf * np.ones_like(bias)
                u_full = -bias
                full_solver = osqp.OSQP()
                full_solver.setup(P=sp.csc_matrix(P_full),
                                    q=q_full,
                                    A=A_full,
                                    l=l_full,
                                    u=u_full,
                                    warm_start=False,
                                    verbose=False)
                
                res_full = full_solver.solve()
                if res_full.info.status != 'solved':
                    continue
                full_sol = res_full.x
                candidate_u0 = full_sol[:u_dim]
                candidate_score = np.linalg.norm(candidate_u0 - action)

            if candidate_score < best_score:
                best_score = candidate_score
                best_u0    = candidate_u0

        if best_u0 is None:
            best_u0 = self.backup(original_state)
            if np.allclose(best_u0, action):
                print("backup equal")
            shielded = False

        self.saved_state = original_state
        self.saved_action = best_u0
        self.shielded = shielded
        return best_u0, shielded

    def __call__(self, state: np.ndarray) -> np.ndarray:
        if self.saved_state is not None and np.allclose(state, self.saved_state):
            return self.saved_action, self.shielded
        return self.solve(state)

    def unsafe(self,
               state: np.ndarray,
               action: np.ndarray) -> bool:
        res = self.solve(state, action=action)[0]
        return not np.allclose(res, action)


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
        env: MarsE2cModel,
        state_space: gym.Space,
        ori_state_space: gym.Space,
        action_space: gym.Space,
        horizon: int,
        unsafe_polys: List[np.ndarray],
        safe_polys: List[np.ndarray], 
        transform=lambda x: x,

        cbf_gamma: float = 0.7
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
        self.cbf_gamma = cbf_gamma
        

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
        self.gamma = 0.5
        
                # --- Placeholders for the pre-computed model and solver ---
        self.solver= None
        self.precomputed = {}
        self.is_model_updated = False

        # --- For caching results ---
        self.saved_state = None
        self.saved_action = None
        self.shielded = False

    def update_model(self):
        """
        Updates the shield with a new dynamics model and performs all expensive,
        state-independent pre-computations. This method should be called whenever
        the fixed Koopman model (A, B, c) or the error bound (eps) changes.
        """
        H_max = self.horizon
        
        z_init = self.ori_state_space.sample()
        a_pi_init = np.zeros(self.u_dim)
        mat_dyn, eps = self.env.get_matrix_at_point(np.concatenate((z_init, a_pi_init)), self.s_dim)
        A, B, c = mat_dyn[:, :self.s_dim], mat_dyn[:, self.s_dim:-1], mat_dyn[:, -1]
        eps_vec = np.full(self.s_dim, float(eps)) if np.isscalar(eps) else np.asarray(eps, float).reshape(-1,)
        # --- 1. Pre-compute matrix powers and affine terms ---
        A_pows = [np.eye(self.s_dim)]
        for _ in range(1, H_max + 1):
            A_pows.append(A_pows[-1] @ A)

        C_list = [np.zeros(self.s_dim)]
        for j in range(1, H_max + 1):
            C_list.append(C_list[-1] + A_pows[j - 1] @ c)

        # --- 2. Pre-compute face info (including relative degrees) ---
        # NOTE: This implementation uses the first polyhedron for shielding.
        poly = self.safe_polys[0]
        P_sel, b_sel = poly[:, :-1].astype(float), poly[:, -1].astype(float)
        faces = [(P_sel[i, :], float(b_sel[i])) for i in range(P_sel.shape[0])]

        def rel_degree(p: np.ndarray) -> Optional[int]:
            M = B.copy()
            for r in range(1, H_max + 1):
                if np.linalg.norm(p @ M, ord=np.inf) > 1e-10: return r
                M = A @ M
            return None

        face_info = [(p, b, rel_degree(p)) for (p, b) in faces]
        r_vals = [r for (_, _, r) in face_info if r is not None]
        H_trap_all = max(r_vals) if r_vals else 1

        # --- 3. Pre-compute tightening terms using prefix sums ---
        tighten_pref = {}
        for p, _, _ in face_info:
            key = tuple(p)
            vals = [np.abs(p @ A_pows[ell]) @ eps_vec for ell in range(H_max)]
            tighten_pref[key] = np.cumsum(vals)

        # --- 4. Build the global, state-independent constraint matrices ---
        G_rows, M_h_rows, v_h_rows = [], [], []
        row_ptrs = [[] for _ in range(H_max)]

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

        # --- 5. Setup the OSQP solver once ---
        P_blocks = [sp.eye(self.u_dim, format="csc")] + [1e-4 * sp.eye(self.u_dim, format="csc")] * (H_max - 1)
        Pmat = sp.block_diag(P_blocks, format="csc")
        lb_actions = np.tile(self.action_space.low, H_max)
        ub_actions = np.tile(self.action_space.high, H_max)
        A_qp = sp.vstack([G_all, sp.eye(H_max * self.u_dim, format="csc")], format="csc")
        
        self.solver = osqp.OSQP()
        self.solver.setup(P=Pmat, q=np.zeros(H_max * self.u_dim), A=A_qp, 
                          l=np.hstack([-np.inf * np.ones(G_all.shape[0]), lb_actions]), 
                          u=np.hstack([np.zeros(G_all.shape[0]), ub_actions]), 
                          verbose=False, polish=False)

        self.precomputed = {
            "M_h": M_h, "v_h": v_h, "row_ptrs": row_ptrs, 
            "H_trap_all": H_trap_all, "ub_actions": ub_actions
        }
        self.is_model_updated = True

    def solve(
        self,
        state: np.ndarray,
        action: Optional[np.ndarray] = None,
        debug: bool = False,
    ) -> Tuple[np.ndarray, bool]:
        """
        Solves for a safe action using the pre-computed model. This method is
        lightweight and designed for real-time execution.
        """
        if not self.is_model_updated:
            # Lazy update: if the model has never been set, initialize it from the env.
            # This is a convenience for the first call.
            print("Model not initialized. Performing first-time update from environment.")
            self.update_model()

        # --- 1. State-Dependent Calculations ---
        z = self.transform(state.reshape(1, -1)).reshape(-1,)
        a_pi = np.zeros(self.u_dim, dtype=float) if action is None else np.asarray(action, float)
        h_all = self.precomputed['M_h'] @ z + self.precomputed['v_h']

        # --- 2. Binary Search for the Largest Feasible Horizon ---
        data = self.precomputed
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
            
            self.solver.update(q=q_new, u=u_new)
            res = self.solver.solve()

            if res.info.status == "solved":
                bestH, best_u0, best_dev = mid, res.x[:self.u_dim], float(np.linalg.norm(res.x[:self.u_dim] - a_pi))
                lo = mid + 1
            else:
                hi = mid - 1

        # --- 3. Return Result or Fallback ---
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
        return act, shielded

    def report(self) -> Tuple[int, int]:
        return self.shield_times, self.agent_times, self.backup_times, self.total_time

    def reset_count(self):
        self.shield_times = 0
        self.agent_times = 0
        self.backup_times = 0
        self.total_time = 0

