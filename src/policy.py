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

        # OSQP solvers for warm-starting
        # self._fixed_solver = osqp.OSQP()
        # self._full_solver = osqp.OSQP()
        self._backup_qp = osqp.OSQP()

    def backup(self, state: np.ndarray) -> np.ndarray:
        """
        Choose a backup action if the projection fails, by finding the
        minimum‐norm u that pushes you out of each unsafe poly, then picking
        the smallest among those.
        """
        s_dim = self.state_space.shape[0]
        # lift & flatten
        with torch.no_grad():
            z = self.transform(state.reshape(1, -1)).reshape(-1,)
        
        # build Hessian = I  (minimize ½‖u‖²)
        P = sp.eye(s_dim, format='csc')
        q = np.zeros(s_dim)

        best_val = np.inf
        best_proj = np.zeros(s_dim)

        for unsafe_mat in self.unsafe_polys:
            A_ineq = unsafe_mat[:, :-1]         # shape (m, s_dim)
            b_ineq = -unsafe_mat[:, -1] - (A_ineq @ z)
            
            # OSQP wants:  l ≤ A_ineq u ≤ u,  so we set l = -∞, u = b_ineq
            A_qp = sp.csc_matrix(A_ineq)
            l_qp = -np.inf * np.ones_like(b_ineq)
            u_qp = b_ineq

            # (re)setup & solve
            backup_qp = osqp.OSQP()
            backup_qp.setup(P=P, q=q, A=A_qp,
                                  l=l_qp, u=u_qp,
                                  warm_start=True,
                                  verbose=False)
            res = backup_qp.solve()
            if res.info.status != 'solved':
                print("unsolved")
                continue

            u_star = res.x
            val = np.linalg.norm(u_star)
            if val < best_val:
                best_val = val
                best_proj = u_star

        # normalize direction
        best_proj /= np.linalg.norm(best_proj)

        # now compute the full‐horizon m vector and call linprog as before
        u_dim = self.action_space.shape[0]
        point = np.concatenate((z, np.zeros(u_dim)))
        mat, _ = self.env.get_matrix_at_point(point, s_dim)
        A_lin = mat[:, :s_dim]
        B_lin = mat[:, s_dim:-1]

        m = np.zeros(self.horizon * u_dim)
        for i in range(self.horizon):
            A_pow = np.linalg.matrix_power(A_lin, self.horizon - i - 1)
            m[i*u_dim:(i+1)*u_dim] = (B_lin.T @ A_pow.T @ (-best_proj)).T

        act_bounds = np.stack((self.action_space.low,
                                self.action_space.high), axis=1)
        bounds = np.tile(act_bounds, (self.horizon, 1))

        # linprog minimizes, so pass -m
        sol = scipy.optimize.linprog(-m, bounds=bounds)
        return sol.x[:u_dim]


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
            # print()
            # Skip if the current state is not in the safe polytope.
            if not np.all(np.dot(P_poly, state) + b_poly <= 0.0):
                continue
                # pass
            
            # === Build safety constraints over the horizon ===
            # We create arrays F, G, and h that capture the propagation of the safe constraints.
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
                    # eps is an interval; we take the maximum possible deviation.
                    # epsmax = np.dot(np.abs(F[j-1][t+1]), eps)
                    epsmax = np.dot(np.abs(F[j-1][t+1]), eps)
                    h[j-1][t] = np.dot(F[j-1][t+1], c) + h[j-1][t+1] + epsmax
                    # h[j-1][t] = np.dot(F[j-1][t+1], c) + h[j-1][t+1]

            # === Assemble the full QP constraints ===
            # Total number of constraints: safety constraints plus action bounds.
            n_constraints = self.horizon * P_poly.shape[0] + 2 * self.horizon * u_dim
            total_vars = self.horizon * u_dim  # full decision vector: [u0; u1; ...; u_{H-1}]
            M = np.zeros((n_constraints, total_vars))
            bias = np.zeros(n_constraints)
            ind = 0
            step = P_poly.shape[0]
            for j in range(self.horizon):
                # Extend G[j] with zeros so that its length is self.horizon.
                G[j] += [np.zeros((P_poly.shape[0], u_dim))] * (self.horizon - j - 1)
                # Concatenate the matrices for time step j (excluding the final entry)
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

            # --- Fixed-QP: u1..u_{H-1} ---
            fixed_total = (self.horizon - 1) * u_dim
            M_first = M[:, :u_dim]
            M_rest  = M[:, u_dim:]
            new_bias = bias + M_first @ action

            P_fixed = (1e-6 * np.eye(fixed_total))
            q_fixed = np.zeros(fixed_total)
            G_fixed = sp.csc_matrix(M_rest)
            h_fixed = -new_bias
            l_fixed = -np.inf * np.ones_like(h_fixed)
            u_fixed = h_fixed

            # Solve fixed-QP with OSQP
            
            fixed_solver = osqp.OSQP()
            fixed_solver.setup(P=sp.csc_matrix(P_fixed),
                                     q=q_fixed,
                                     A=G_fixed,
                                     l=l_fixed,
                                     u=u_fixed,
                                     warm_start=False,
                                     verbose=False,
                                    #  max_iter = 8000,
                                )
            res_fixed = fixed_solver.solve()
            fixed_feasible = (res_fixed.info.status == 'solved')
            fixed_feasible = False

            if fixed_feasible:
                candidate_u0 = action.copy()
                candidate_score = 0.0
                shielded = False
            else:
                # --- Full-QP: optimize [u0; u1..] ---
                total_vars = self.horizon * u_dim
                P_full = 1e-6 * np.eye(total_vars)
                # keep u0 near policy
                P_full[:u_dim, :u_dim] = np.eye(u_dim)
                q_full = -np.concatenate((action, np.zeros((self.horizon - 1)*u_dim)))
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
            unsafe_policy = None):
        self.shield = shield_policy
        self.agent = unsafe_policy
        self.shield_times = 0
        self.backup_times = 0
        self.agent_times = 0
        self.total_time = 0.

    def __call__(self, state: np.ndarray, action: np.ndarray = None, **kwargs) -> np.ndarray:
        start = time.time()
        if action is not None:
            proposed_action = action
        else:
            proposed_action = self.agent(state, **kwargs)

        
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

