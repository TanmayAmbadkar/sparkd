import gymnasium as gym
import numpy as np
from typing import Optional, List, Tuple
import cvxopt
import scipy
import torch
import time

from pytorch_soft_actor_critic.sac import SAC
from pytorch_soft_actor_critic.replay_memory import ReplayMemory
from ppo import PPO
from scipy.optimize import linprog
from e2c.env_model import MarsE2cModel


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
    """
    Wrap an underlying policy in a safety layer based on prejection onto a
    weakest precondition.
    """

    def __init__(self,
                 env: MarsE2cModel,
                 state_space: gym.Space,
                 ori_state_space: gym.Space,
                 action_space: gym.Space,
                 horizon: int,
                 unsafe_polys: List[np.ndarray],
                 safe_polys: List[np.ndarray], 
                 transform = lambda x: x):
        self.env = env
        self.horizon = horizon
        self.state_space = state_space
        self.ori_state_space = ori_state_space
        self.action_space = action_space
        self.unsafe_polys = unsafe_polys
        self.safe_polys = safe_polys
        self.saved_state = None
        self.saved_action = None
        self.transform = transform
        self.mat_dyn = None
        self.F = None
        self.G = None
        self.h = None

        
        self.slack = 0.1
        
    def backup(self, state: np.ndarray) -> np.ndarray:
        """
        Choose a backup action if the projection fails.
        """
        s_dim = self.state_space.shape[0]
        with torch.no_grad():
            state = self.transform(state.reshape(1, -1))
        # state = state.numpy()
        state = state.reshape(-1,)

        P = cvxopt.spmatrix(1.0, range(s_dim), range(s_dim))
        q = cvxopt.matrix(0.0, (s_dim, 1))

        best_val = 1e10
        best_proj = np.zeros_like(state)
        for unsafe_mat in self.unsafe_polys:
            tmp = unsafe_mat[:self.ori_state_space.shape[0]*2, :-1]
            G = cvxopt.matrix(tmp)
            h = cvxopt.matrix(-unsafe_mat[:self.ori_state_space.shape[0]*2, -1] - np.dot(tmp, state))
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

    
    def solve(self, state: np.ndarray, action: Optional[np.ndarray] = None, debug: bool = False) -> Tuple[np.ndarray, bool]:
        """
        Solve the safety synthesis problem.
        
        First, try to check whether the policy’s proposed action (u0)
        can be paired with a feasible future action sequence (u1,...,u_H-1)
        that keeps the trajectory safe. This is done by fixing u0 and
        optimizing only over future actions.
        
        If that QP is infeasible, fall back to solving the original QP that
        is allowed to change u0.
        
        Returns:
        (safe_action, shielded) where safe_action is the computed safe u0,
        and shielded is True if we could keep the original action.
        """
        original_state = state.copy()
        shielded = True
        s_dim = self.state_space.shape[0]
        u_dim = self.action_space.shape[0]
        # Transform the state appropriately.
        state = self.transform(state.reshape(1, -1)).reshape(-1,)
        # Use the provided action if available; otherwise, default to zero.
        if action is None:
            action = np.zeros(u_dim)
        
        # Get local dynamics: here, env.get_matrix_at_point returns a matrix which we split into A, B, and c.
        point = np.concatenate((state, action))
        mat_dyn, eps = self.env.get_matrix_at_point(point, s_dim)

        A = mat_dyn[:, :s_dim]
        B = mat_dyn[:, s_dim:-1]
        c = mat_dyn[:, -1]

        
        best_score = 1e10
        best_u0 = None


        # Iterate over each safe polytope.
        for poly in self.safe_polys:
            P_poly = poly[:self.ori_state_space.shape[0]*2, :-1]
            b_poly = poly[:self.ori_state_space.shape[0]*2, -1]
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

            # === First attempt: fix u0 to the given action and optimize over u1,...,u_{H-1} ===
            fixed_total = (self.horizon - 1) * u_dim  # dimension for u1,...,u_{H-1}
            # Partition the full variable: columns [0:u_dim] correspond to u0, [u_dim:] correspond to future actions.
            M_first = M[:, :u_dim]
            M_rest = M[:, u_dim:]
            # Adjust the bias: since u0 is fixed to "action", add M_first * action to the bias.
            new_bias = bias + np.dot(M_first, action)
            # Now, the constraints become: M_rest * x + new_bias <= 0, where x represents [u1; ...; u_{H-1}].
            
            # Set up a simple quadratic objective for the future actions (e.g. minimize ||x||^2).
            P_fixed = 1e-4 * np.eye(fixed_total)
            q_fixed = np.zeros((fixed_total,))
            
            # Also set up bounds for the future actions.
            # act_bounds = np.stack((self.action_space.low, self.action_space.high), axis=1)
            # # Replicate the bounds for (horizon-1) steps.
            # bounds_fixed = np.concatenate([act_bounds] * (self.horizon - 1), axis=0)
            
            # Build inequality constraints for the fixed-QP:
            # The safety constraints: M_rest * x <= -new_bias.
            G_fixed = M_rest
            h_fixed = new_bias
            
            # Convert to cvxopt matrices.
            try:
                sol_fixed = cvxopt.solvers.qp(cvxopt.matrix(P_fixed),
                                            cvxopt.matrix(q_fixed),
                                            cvxopt.matrix(G_fixed),
                                            cvxopt.matrix(-h_fixed))
            except Exception as e:
                sol_fixed = {'status': 'infeasible'}
            fixed_feasible = (sol_fixed['status'] == 'optimal')
            
            if fixed_feasible:
                # The fixed-QP is feasible: the original policy action u0 is safe.
                candidate_u0 = action.copy()
                candidate_score = 0  # no deviation from the policy action.
                shielded = False
            else:
                # Fixed-QP failed: fall back to solving the full QP where u0 is free.
                P_full = 1e-4 * np.eye(total_vars)
                # Add extra weight on the first block so the solution stays close to the policy action.
                P_full[:u_dim, :u_dim] += np.eye(u_dim)
                q_full = -np.concatenate((action, np.zeros((self.horizon - 1) * u_dim)))
                try:
                    sol_full = cvxopt.solvers.qp(cvxopt.matrix(P_full),
                                                cvxopt.matrix(q_full),
                                                cvxopt.matrix(M),
                                                cvxopt.matrix(-bias))
                except Exception as e:
                    sol_full = {'status': 'infeasible'}
                if sol_full['status'] != 'optimal':
                    continue  # try the next safe polytope
                full_sol = np.asarray(sol_full['x']).squeeze()
                candidate_u0 = full_sol[:u_dim]
                candidate_score = np.linalg.norm(candidate_u0 - action)
            
            # Record the best candidate u0 (the one closest to the original action).
            if candidate_score < best_score:
                best_score = candidate_score
                best_u0 = candidate_u0

        # If no safe solution was found across any polytope, use the backup action.
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
            shielded = "NEURAL"
            self.agent_times += 1
        end = time.time()
        self.total_time += end - start
        return proposed_action, shielded

    def report(self) -> Tuple[int, int]:
        return self.shield_times, self.agent_times, self.backup_times, self.total_time

    def reset_count(self):
        self.shield_times = 0
        self.agent_times = 0
        self.backup_times = 0
        self.total_time = 0

