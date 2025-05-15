import gymnasium as gym
import numpy as np
from typing import Optional, List, Tuple
import cvxopt
import scipy
import torch
import time

from .env_model import MARSModel
from pytorch_soft_actor_critic.sac import SAC
from pytorch_soft_actor_critic.replay_memory import ReplayMemory
from scipy.optimize import linprog
import osqp
import scipy.sparse as sp



cvxopt.solvers.options['show_progress'] = False


class SACPolicy:

    def __init__(self,
                 gym_env: gym.Env,
                 replay_size: int,
                 seed: int,
                 batch_size: int,
                 sac_args):
        self.agent = SAC(gym_env.observation_space.shape[0],
                         gym_env.action_space, sac_args)
        self.memory = ReplayMemory(replay_size, seed)
        self.updates = 0
        self.batch_size = batch_size

    def __call__(self, state: np.ndarray, evaluate: bool = False):
        return self.agent.select_action(state, evaluate=evaluate)

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


class ProjectionPolicy:
    """
    Wrap an underlying policy in a safety layer based on prejection onto a
    weakest precondition.
    """

    def __init__(self,
                 env: MARSModel,
                 state_space: gym.Space,
                 action_space: gym.Space,
                 horizon: int,
                 unsafe_polys: List[np.ndarray],
                 safe_polys: List[np.ndarray]):
        self.env = env
        self.horizon = horizon
        self.state_space = state_space
        self.action_space = action_space
        self.unsafe_polys = unsafe_polys
        self.safe_polys = safe_polys
        self.saved_state = None
        self.saved_action = None
        
    def backup(self, state: np.ndarray) -> np.ndarray:
        """
        Choose a backup action if the projection fails.
        """
        s_dim = self.state_space.shape[0]
        P = cvxopt.spmatrix(1.0, range(s_dim), range(s_dim))
        q = cvxopt.matrix(0.0, (s_dim, 1))

        best_val = 1e10
        best_proj = np.zeros_like(state)
        for unsafe_mat in self.unsafe_polys:
            tmp = unsafe_mat[:, :-1]
            G = cvxopt.matrix(tmp)
            h = cvxopt.matrix(-unsafe_mat[:, -1] - np.dot(tmp, state))
            sol = cvxopt.solvers.qp(P, q, G, h)
            soln = np.asarray(sol['x']).squeeze()
            if len(soln.shape) == 0:
                soln = soln[None]
            val = np.linalg.norm(soln)
            if val < best_val:
                best_val = val
                best_proj = soln
        best_proj = best_proj / np.linalg.norm(best_proj)
        u_dim = self.action_space.shape[0]
        point = np.concatenate((state, np.zeros(u_dim)))
        mat, _ = self.env.get_matrix_at_point(point, s_dim)
        # M is the concatenated linear model, so we need to split it into the
        # dynamics and the input
        A = mat[:, :s_dim]
        B = mat[:, s_dim:-1]
        # c = mat[:, -1]   # c only contributes a constant and isn't needed
        # Some analysis:
        # x_{i+1} = A x_i + B u_i + c
        # x_1 = A x_0 + B u_0 + c
        # x_2 = A (A x_0 + B u_0 + c) + B u_1 + c
        #     = A^2 x_0 + A B u_0 + A c + B u_1 + c
        # x_3 = A (A^2 x_0 + A B u_0 + A c + B u_1 + c) + B u_2 + c
        #     = A^3 x_0 + A^2 B u_0 + A^2 c + A B u_1 + A c + B u_2 + c
        # x_i = A^i x_0 + A^{i-1} B u_0 + ... + A B u_{i-2} + B u_{i-1} +
        #           A^{i-1} c + ... + A c + c
        # x_H = A^H x_0
        #     + \sum_{i=0}^{H-1} A^{H-i-1} B u_i
        #     + \sum_{i=1}^{H-1} A^i c
        # Now we maximize -best_proj^T (x_H - x_0). -best_proj^T x_0 is
        # constant so we can igonore it and just maximize -best_proj^T x_H.
        # (let q = -best_proj for convenience)
        #   q^T x_H
        # = q^T (A^H x_0 + sum A^i c + sum A^{H-i-1} B u_i)
        # = q^T A^H x_0 + q^T sum A^i c + q^T sum A^{H-i-1} B u_i
        # We can remove the constants q^T A^H x_0 and q^T sum A^i c
        # maximize q^T sum A^{H-i-1} B u_i
        #    = sum q^T A^{H-i-1} B u_i
        #    = sum (q^T A^{H-i-1} B) u_i
        #    = sum ((A^{H-i-1} B)^T q)^T u_i
        # So in the end, let
        # m = [((A^{H-1} B)^T q)^T
        #      ((A^{H-2} B)^T q)^T
        #      ...
        #      ((A B)^T q)^T
        #      (B^T q)^T]
        # and let u = [u_0 u_1 ... u_{H-1}]. Then we need to solve
        # maximize m^T u subject to action space constraints
        m = np.zeros(self.horizon * u_dim)
        for i in range(self.horizon):
            m[u_dim*i:u_dim*(i+1)] = \
                np.dot(np.dot(np.linalg.matrix_power(A, self.horizon - i - 1),
                              B).T, -best_proj).T
        act_bounds = np.stack((self.action_space.low, self.action_space.high),
                              axis=1)
        bounds = np.concatenate([act_bounds] * self.horizon, axis=0)
        # linprog minimizes, so we need -m here
        res = scipy.optimize.linprog(-m, bounds=bounds)
        # Return the first action
        return res['x'][:u_dim]


    def solve(self,    # noqa: C901
              state: np.ndarray,
              action: Optional[np.ndarray] = None,
              debug: bool = False) -> np.ndarray:
        """
        Solve the synthesis problem and store the result. This sets the saved
        action and state because very often we will call unsafe and then
        __call__ on the same state.
        """
        
        shielded = True
        s_dim = self.state_space.shape[0]
        u_dim = self.action_space.shape[0]
        # If we don't have a proposed action, look for actions with small
        # magnitude
        if action is None:
            action = np.zeros(u_dim)
        # Get the local dynamics
        point = np.concatenate((state, action))
        mat, eps = self.env.get_matrix_at_point(point, s_dim)
        A = mat[:, :s_dim]
        B = mat[:, s_dim:-1]
        c = mat[:, -1]

        best_score = 1e10
        best_u0 = None

        for poly in self.safe_polys:
            P_poly = poly[:, :-1]
            b_poly = poly[:, -1]
            # Skip if the current state is not in the safe polytope.
            if not np.all(np.dot(P_poly, state) + b_poly <= 0.0):
                continue

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
                    epsmax = np.dot(np.abs(F[j-1][t+1]), eps)
                    h[j-1][t] = np.dot(F[j-1][t+1], c) + h[j-1][t+1] + epsmax

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

        
            # Fixed-QP failed: fall back to solving the full QP where u0 is free.
            P_full = 1e-4 * np.eye(total_vars)
            # Add extra weight on the first block so the solution stays close to the policy action.
            P_full[:u_dim, :u_dim] += np.eye(u_dim)
            q_full = -np.concatenate((action, np.zeros((self.horizon - 1) * u_dim)))
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

            
            # Record the best candidate u0 (the one closest to the original action).
            if candidate_score < best_score:
                best_score = candidate_score
                best_u0 = candidate_u0

        if best_u0 is None:
            best_u0 = self.backup(state)
            shielded = False
            # best_u0 = action

        self.saved_state = state
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
            unsafe_policy: SACPolicy):
        self.shield = shield_policy
        self.agent = unsafe_policy
        self.shield_times = 0
        self.backup_times = 0
        self.agent_times = 0
        self.total_time = 0.

    def __call__(self, state: np.ndarray, **kwargs) -> np.ndarray:
        start = time.time()
        proposed_action = self.agent(state, **kwargs)
        if self.shield.unsafe(state, proposed_action):
            act, shielded  = self.shield(state)
            self.shield_times += 1 if shielded else 0
            self.backup_times += 1 if not shielded else 0
            
            return act
        self.agent_times += 1
        end = time.time()
        self.total_time += end - start
        return proposed_action

    def report(self) -> Tuple[int, int]:
        return self.shield_times, self.agent_times, self.backup_times, self.total_time

    def reset_count(self):
        self.shield_times = 0
        self.agent_times = 0
        self.backup_times = 0
        self.total_time = 0


class CSCShield:
    """
    Construct a shield from a neural policy and a conservative safety critic.
    """

    def __init__(self, policy: SACPolicy, cost_model: torch.nn.Module,
                 threshold: float = 0.2):
        self.policy = policy
        self.cost_model = cost_model
        self.threshold = threshold

    def __call__(self, state: np.ndarray, **kwargs) -> np.ndarray:
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(self.policy(state, **kwargs),
                              dtype=torch.float32)
        iters = 0
        best_action = action
        score = self.cost_model(torch.cat((state, action)))
        best_score = score
        while score > self.threshold and iters < 100:
            action = torch.tensor(self.policy(state, **kwargs),
                                  dtype=torch.float32)
            score = self.cost_model(torch.cat((state, action)))
            if score < best_score:
                best_score = score
                best_action = action
            iters += 1
        return best_action.detach().numpy()

    def report(self) -> Tuple[int, int]:
        return 0, 0

    def reset_count(self):
        pass