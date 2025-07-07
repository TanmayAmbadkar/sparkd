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


class Shield:
    """
    Construct a shield from a neural policy and a safety layer.
    """

    def __init__(
            self,
            shield_policy: ProjectionPolicy,
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
        return act, shielded

    def report(self) -> Tuple[int, int]:
        return self.shield_times, self.agent_times, self.backup_times, self.total_time

    def reset_count(self):
        self.shield_times = 0
        self.agent_times = 0
        self.backup_times = 0
        self.total_time = 0

