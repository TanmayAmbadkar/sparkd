import gymnasium as gym
import numpy as np
from typing import Optional, List, Tuple, Union
import scipy
import torch
import time

from pytorch_soft_actor_critic.sac import SAC
from pytorch_soft_actor_critic.replay_memory import ReplayMemory
from ppo import PPO
from koopman.env_model import KoopmanLinearModel, FixedLinearModel
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

        # --- NEW: Attributes for pre-computation cache ---
        self._A, self._B, self._c, self._eps = None, None, None, None
        self._precomputed_F = {}
        self._precomputed_h_const = {}
        self._precomputed_M = {}
        
        # --- NEW: Attributes for stateful call optimization ---
        self.saved_state = None
        self.saved_action = None
        self.shielded = None

    def update_model(self):
        """
        Pre-computes and caches all state-independent matrices for the QP.

        This is the main optimization. This method performs the expensive,
        iterative matrix propagations once and stores the results. The `solve`
        method can then run much faster by reusing these cached matrices.

        Call this method whenever the underlying linear dynamics model changes.

        Args:
            A: State transition matrix from the linearization.
            B: Control matrix from the linearization.
            c: Constant offset vector from the linearization.
            eps: Error bounds from the linearization.
        """
        print("Pre-computing safety projection matrices...")
        mat_dyn, eps = self.env.get_matrix_at_point(None, self.s_dim)
        A, B, c = mat_dyn[:, :self.s_dim], mat_dyn[:, self.s_dim:-1], mat_dyn[:, -1]
        self._A, self._B, self._c = A, B, c
        self._eps = np.full(self.s_dim, float(eps)) if np.isscalar(eps) else np.asarray(eps, float).reshape(-1,)
        s_dim = self.state_space.shape[0]
        u_dim = self.action_space.shape[0]

        # Clear old cache
        self._precomputed_F.clear()
        self._precomputed_h_const.clear()
        self._precomputed_M.clear()

        # Pre-compute for each safe polytope
        for poly_idx, poly in enumerate(self.safe_polys):
            P_poly, b_poly = poly[:, :-1], poly[:, -1]

            # --- Pre-compute F and h_const (state-independent parts) ---
            F = []
            h_const = []
            for j in range(1, self.horizon + 1):
                F.append([None] * (j + 1))
                h_const.append([None] * (j + 1))
                F[j - 1][j] = P_poly
                h_const[j - 1][j] = b_poly
                for t in range(j - 1, -1, -1):
                    F[j - 1][t] = np.dot(F[j - 1][t + 1], self._A)
                    epsmax = np.dot(np.abs(F[j - 1][t + 1]), self._eps)
                    h_const[j - 1][t] = np.dot(F[j - 1][t + 1], self._c) + h_const[j - 1][t + 1] + epsmax
            
            self._precomputed_F[poly_idx] = F
            self._precomputed_h_const[poly_idx] = h_const

            # --- Pre-compute G and M (also state-independent) ---
            G = []
            for j in range(1, self.horizon + 1):
                G.append([None] * (j + 1))
                G[j - 1][j] = np.zeros((b_poly.shape[0], u_dim))
                for t in range(j - 1, -1, -1):
                    G[j - 1][t] = np.dot(F[j - 1][t + 1], self._B)
            
            total_vars = self.horizon * u_dim
            n_safety_constraints = self.horizon * P_poly.shape[0]
            M = np.zeros((n_safety_constraints, total_vars))
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
        Solves the safety projection QP using pre-computed matrices.
        """
        original_state = state.copy()
        shielded = True
        u_dim = self.action_space.shape[0]
        state = self.transform(state.reshape(1, -1)).reshape(-1,)
        if action is None:
            action = np.zeros(u_dim)
            
        if self._A is None:
            raise RuntimeError("Must call .update() before .solve() to pre-compute matrices.")

        best_score = np.inf
        best_u0 = None

        for poly_idx, poly in enumerate(self.safe_polys):
            P_poly, b_poly = poly[:, :-1], poly[:, -1]
            
            if not np.all(np.dot(P_poly, state) + b_poly <= 0.0):
                continue
            
            # --- Retrieve pre-computed matrices ---
            F = self._precomputed_F[poly_idx]
            h_const = self._precomputed_h_const[poly_idx]
            M_safety = self._precomputed_M[poly_idx]

            # --- Assemble full QP constraint matrix M ---
            n_action_constraints = 2 * self.horizon * u_dim
            n_constraints = M_safety.shape[0] + n_action_constraints
            total_vars = self.horizon * u_dim
            
            M = np.zeros((n_constraints, total_vars))
            M[:M_safety.shape[0], :] = M_safety
            
            # Add action bounds constraints to M
            action_eye = np.eye(total_vars)
            M[M_safety.shape[0]:M_safety.shape[0] + total_vars, :] = action_eye
            M[M_safety.shape[0] + total_vars:, :] = -action_eye
            
            # --- Calculate the state-dependent bias vector ---
            bias = np.zeros(n_constraints)
            ind = 0
            step = P_poly.shape[0]
            for j in range(self.horizon):
                bias[ind:ind+step] = h_const[j][0] + np.dot(F[j][0], state)
                ind += step
            
            bias[ind:ind+total_vars] = -np.tile(self.action_space.high, self.horizon)
            bias[ind+total_vars:] = np.tile(self.action_space.low, self.horizon)

            # --- LP Feasibility Check (Identical logic, but faster setup) ---
            fixed_total = (self.horizon - 1) * u_dim
            M_first = M[:, :u_dim]
            M_rest = M[:, u_dim:]
            new_bias = bias + M_first @ action

            res_lp = scipy.optimize.linprog(c=np.zeros(fixed_total),
                                            A_ub=M_rest,
                                            b_ub=-new_bias,
                                            method='highs',
                                            bounds=(self.action_space.low[0], self.action_space.high[0]))
            fixed_feasible = res_lp.success

            if fixed_feasible:
                candidate_u0 = action.copy()
                candidate_score = 0.0
                shielded = False
            else:
                # --- Full-QP optimization (Identical logic) ---
                P_full = 1e-6 * np.eye(total_vars)
                P_full[:u_dim, :u_dim] = np.eye(u_dim)
                q_full = np.zeros(total_vars)
                q_full[:u_dim] = -action
                
                full_solver = osqp.OSQP()
                full_solver.setup(P=sp.csc_matrix(P_full), q=q_full,
                                  A=sp.csc_matrix(M), l=-np.inf * np.ones_like(bias), u=-bias,
                                  verbose=False)
                
                res_full = full_solver.solve()
                if res_full.info.status != 'solved':
                    continue
                candidate_u0 = res_full.x[:u_dim]
                candidate_score = np.linalg.norm(candidate_u0 - action)

            if candidate_score < best_score:
                best_score = candidate_score
                best_u0 = candidate_u0

        if best_u0 is None:
            best_u0 = self.backup(original_state)
            shielded = False # Backup implies the original action was unsafe/infeasible

        self.saved_state = original_state
        self.saved_action = best_u0
        self.shielded = shielded
        return best_u0, shielded
    
    # The __call__, unsafe, and backup methods remain unchanged
    # ... (Copy the __call__, unsafe, and backup methods from your original code here) ...
    def __call__(self, state: np.ndarray) -> np.ndarray:
        if self.saved_state is not None and np.allclose(state, self.saved_state):
            return self.saved_action, self.shielded
        return self.solve(state)

    def unsafe(self,
               state: np.ndarray,
               action: np.ndarray) -> bool:
        res = self.solve(state, action=action)[0]
        return not np.allclose(res, action)

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
        A_lin = self._A
        B_lin = self._B
        
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
        env: Union[KoopmanLinearModel, FixedLinearModel],
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
        return act, shielded, np.linalg.norm(act - proposed_action)

    def report(self) -> Tuple[int, int]:
        return self.shield_times, self.agent_times, self.backup_times, self.total_time

    def reset_count(self):
        self.shield_times = 0
        self.agent_times = 0
        self.backup_times = 0
        self.total_time = 0

