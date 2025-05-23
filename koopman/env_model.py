from typing import Optional, List, Callable
import numpy as np
import scipy.stats
from koopman.network import KoopmanLightning, fit_koopman
from abstract_interpretation.verification import get_constraints, get_ae_bounds, get_variational_bounds
from abstract_interpretation import domains
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import explained_variance_score, r2_score
from typing import Union
import seaborn as sns
import matplotlib.pyplot as plt


class MarsE2cModel:
    """
    A model that uses the KoopmanLightning to obtain A, B, and c matrices
    and provides a similar interface to MARSModel.
    """
    def __init__(self, koopman_model: KoopmanLightning, s_dim=None, original_s_dim = None):
        self.koopman_model = koopman_model
        self.s_dim = s_dim
        self.original_s_dim = original_s_dim
        self.error = 0

    def __call__(self, state, action,  normalized: bool = False) -> np.ndarray:
        """
        Predict the next state given the current state x and action u.
        """
        
        x_norm = state
        u_norm = action
        # Convert to tensors
        x_tensor = torch.tensor(x_norm, )
        u_tensor = torch.tensor(u_norm, )

        # print(x_tensor.shape, u_tensor.shape)
        # Use KoopmanLightning to predict next state
        z_t_next = self.koopman_model(x_tensor, u_tensor)

        # Predict next latent state
        
        return z_t_next.detach().cpu().numpy()

    def get_matrix_at_point(self, point: np.ndarray, s_dim: int, steps: int = 1, normalized: bool = False):
        """
        Get the linear model at a particular point.
        Returns M and eps similar to the original MARSModel.
        M is such that the model output can be approximated as M @ [x; 1],
        where x is the input state-action vector.

        Parameters:
        - point: The concatenated (state, action) input vector of length s_dim + u_dim.
        - s_dim: The dimension of the state (and latent dimension, if they match).
        - steps: Number of steps to unroll for error estimation (not used here).
        - normalized: Whether 'point' is already normalized (not used here).

        Returns:
        - M: The linear approximation matrix of shape [s_dim, (s_dim + u_dim + 1)].
        - eps: A vector of length s_dim, taken from diag(A_t @ A_t^T).
        """

        # 1. If needed, unnormalize:
        # if not normalized:
        #     point = (point - self.inp_means) / self.inp_stds

        # 2. Split into state (x_norm) and action (u_norm))
        x_norm = point[:s_dim]
        u_norm = point[s_dim:]

        # 3. Convert to torch tensors
        x_tensor = torch.Tensor(x_norm).unsqueeze(0)
        u_tensor = torch.Tensor(u_norm).unsqueeze(0)

        # 4. Run the E2C transition:
        #    Returns (z_next, z_next_mean, A_t, B_t, c_t, v_t, r_t)

        with torch.no_grad():
            z_next, z_next_mean, A_t, B_t, c_t, v_t, r_t = self.koopman_model.transition(
                x_tensor, x_tensor, u_tensor
            )

        # 5. Convert PyTorch tensors to NumPy, remove batch dimension
        A_t = A_t.detach().cpu().numpy()   # shape [s_dim, s_dim]
        B_t = B_t.detach().cpu().numpy()    # shape [s_dim, u_dim]C
        c_t = c_t.squeeze(0).detach().cpu().numpy()    # shape [s_dim]
        # c_T = np.zeros(A_t.shape[0]) 

        # 6. Construct M by stacking [A | B | c], giving shape [s_dim, s_dim + u_dim + 1]
        #    Note: c_t[:, None] is the bias column
        M = np.hstack((A_t, B_t, c_t[:, None]))

        # 7. Compute eps as the diagonal of A_t @ A_t^T.
        #    That yields a 1D array of length s_dim.
        # eps = np.diag(A_tA_tT) # shape [s_dim]
        # eps = np.zeros_like(c_t)
        # eps = self.koopman_model.get_eps(x_tensor, u_tensor).squeeze(0).detach().cpu().numpy()
        # print("Eps:", eps)
        eps = self.error
        return M, eps



    def __str__(self):
        return "MarsE2cModel using KoopmanLightning"


class EnvModel:
    """
    A full environment model including a symbolic model and a neural model.

    This model includes a symbolic (MARS) model of the dynamics, a neural
    model which accounts for dynamics not captured by the symbolic model, and a
    second neural model for the reward function.
    """

    def __init__(
            self,
            mars: MarsE2cModel,
            observation_space_low,
            observation_space_high):
        """
        Initialize an environment model.

        Parameters:
        mars - A symbolic model.
        net - A neural model for the residuals.
        reward - A neural model for the reward.
        """
        self.mars = mars
        self.observation_space_low = np.array(observation_space_low)
        self.observation_space_high = np.array(observation_space_high)
        

    def __call__(self,
                 state: np.ndarray,
                 action: np.ndarray,
                 use_neural_model: bool = True) -> np.ndarray:
        """
        Predict a new state and reward value for a given state-action pair.

        Parameters:
        state (1D array) - The current state of the system.
        action (1D array) - The action to take

        Returns:
        A tuple consisting of the new state and the reward.
        """
        state = state.reshape(-1, )
        action = action.reshape(-1, )
        inp = np.concatenate((state, action), axis=0)
        symb = self.mars(inp)
        
            
        return np.clip(symb[0], self.observation_space_low, self.observation_space_high), 0

    def get_symbolic_model(self) -> MarsE2cModel:
        """
        Get the symbolic component of this model.
        """
        return self.mars

    def get_confidence(self) -> float:
        return self.confidence

    @property
    def error(self) -> float:
        return self.mars.error




def get_environment_model(     # noqa: C901
        input_states: np.ndarray,
        actions: np.ndarray,
        output_states: np.ndarray,
        rewards: np.ndarray,
        domain,
        seed: int = 0,
        data_stddev: float = 0.01,
        latent_dim: int = 4,
        horizon: int = 5,
        koopman_model = None,
        epochs: int = 50) -> EnvModel:

    
    # means = np.mean(input_states, axis=0)
    # stds = np.std(input_states, axis=0)
    # stds[np.equal(np.round(stds, 1), np.zeros(*stds.shape))] = 1

    # print("Means:", means)
    # print("Stds:", stds)
    
    means = np.zeros(domain.lower.shape[1])
    stds = np.ones(domain.lower.shape[1])
    
    # if koopman_model is not None:
    #     means = koopman_model.mean + 0.001 * (means - koopman_model.mean)
    #     stds = koopman_model.std + 0.001 * (means - koopman_model.mean)
    
    # domain.lower = (domain.lower - means) / stds
    # domain.upper = (domain.upper - means) / stds
    # print("Input states:", input_states)
    
    if koopman_model is None:
        koopman_model = KoopmanLightning(input_states.shape[-1], latent_dim, actions.shape[-1], horizon)
        koopman_model_copy = KoopmanLightning(input_states.shape[-1], latent_dim, actions.shape[-1], horizon)
    else:
        koopman_model_copy = KoopmanLightning(input_states.shape[-1], latent_dim, actions.shape[-1], horizon)
        koopman_model_copy.load_state_dict(koopman_model.state_dict())
        koopman_model_copy.eval()
        
        
    fit_koopman(input_states, actions, output_states, koopman_model, horizon, epochs=epochs)

    koopman_model.mean = means
    koopman_model.std = stds

    
    # print(input_states.shape, actions.shape, output_states.shape)
    # parsed_mars = MarsE2cModel(koopman_model, latent_dim, input_states.shape[-1])
    # X = np.concatenate((input_states.reshape(-1, input_states.shape[-1]), actions.reshape(-1, actions.shape[-1])), axis=1)
    # print(X.shape)
    # Yh = np.array([parsed_mars(state, normalized=True) for state in X]).reshape(-1, input_states.shape[-1])
    # output_states = output_states.reshape(-1, input_states.shape[-1])
    # print("Model estimation error:", np.mean((Yh - (output_states).reshape(-1, input_states.shape[-1]))**2))
    # print("Explained Variance Score:", explained_variance_score(
    #     output_states, Yh))
    

    
    print(input_states.shape, actions.shape, output_states.shape)
    parsed_mars = MarsE2cModel(koopman_model, latent_dim, input_states.shape[-1])
    parsed_mars_copy = MarsE2cModel(koopman_model_copy, latent_dim, input_states.shape[-1])
    koopman_model.horizon = horizon
    Yh = parsed_mars(input_states, actions, normalized=True)
    Yh_copy = parsed_mars_copy(input_states, actions, normalized=True)

    output_states = parsed_mars.koopman_model.transform(output_states)
    output_states_copy = parsed_mars_copy.koopman_model.transform(output_states[:, :, :input_states.shape[-1]])
    # output_states = output_states.reshape(-1, input_states.shape[-1])

    print(np.min(Yh[:,0], axis = 0), np.max(Yh[:,0], axis = 0))
    print(np.min(output_states[:,0], axis = 0), np.max(output_states[:,0], axis = 0))
    ev_score = None
    r2 = None
    ev_score_old = None
    r2_old = None
    for i in range(horizon):
        print(f"Model estimation error, horizon {i}:", np.mean((Yh[:,i, :input_states.shape[-1]] - (output_states[:, i, :input_states.shape[-1]]))**2)) 
        print(f"Model estimation error old, horizon {i}:", np.mean((Yh_copy[:,i, :input_states.shape[-1]] - (output_states_copy[:, i, :input_states.shape[-1]]))**2)) 
        
        ev_score = explained_variance_score(
            output_states[:,i, :input_states.shape[-1]].reshape(-1), Yh[:,i, :input_states.shape[-1]].reshape(-1))
        r2 = r2_score(
            output_states[:,i,  :input_states.shape[-1]].reshape(-1), Yh[:,i, :input_states.shape[-1]].reshape(-1))
        print(f"Explained Variance Score, horizon {i}:", ev_score)
        print(f"R2 Score, horizon {i}:", r2)
        ev_score_old = explained_variance_score(
            output_states_copy[:,i, :input_states.shape[-1]].reshape(-1), Yh_copy[:,i, :input_states.shape[-1]].reshape(-1))
        r2_old = r2_score(
            output_states_copy[:,i,  :input_states.shape[-1]].reshape(-1), Yh_copy[:,i, :input_states.shape[-1]].reshape(-1))
        print(f"Old Explained Variance Score, horizon {i}:", ev_score)
        print(f"Old R2 Score, horizon {i}:", r2)
                
            # number of output dims

    # 1) compute dimension‑wise maximum abs‑error over all samples & timesteps
    #    (broadcast output_states across the middle axis of Yh)
    #    Yh: (n_samples, n_steps, n_out)
    #    output_states: (n_samples, n_out)
            
        # 1) Compute the absolute residuals for every (sample, step, feature)
        
    if ev_score_old > ev_score:
        Yh = Yh_copy
        output_states = output_states_copy
        parsed_mars = parsed_mars_copy
        print("Using old model")
        
    res = np.abs(
        Yh[:, 0] 
        - output_states[:, 0]
    )           # shape: (N, T, n_out)

    sns.boxplot(data=res.reshape(-1, output_states.shape[-1]))
    plt.savefig("boxplot.png")
    plt.close()
    # 2) Flatten over samples & time
    res_flat = res.reshape(-1, output_states.shape[-1])             # shape: (N*T, n_out))
    print("Flattened res", res_flat.shape)

    # 3) Empirical max

    # 4) Empirical 95%-quantile (or whatever α you choose)
    quantile = np.percentile(res_flat, 99, axis=0)   # shape: (n_out,)

    q1 = np.percentile(res_flat, 25, axis=0)   # shape: (n_out,)
    q3 = np.percentile(res_flat, 75, axis=0)   # shape: (n_out,)
    print("Q1", q1)
    print("Q3", q3)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    print("lower_bound", lower_bound)
    print("upper_bound", upper_bound)
    print("Below LB", res_flat[res_flat < lower_bound].shape)
    print("Above UB", res_flat[res_flat > upper_bound].shape)
    print(quantile)
    print(upper_bound)
    
    print("Max error:", upper_bound)

    # 5) still store the full vector
    # parsed_mars.error = upper_bound
    
    #  Get the maximum distance between a predction and a datapoint
    diff = np.amax(np.abs( Yh[:, 0] 
        - output_states[:, 0, :]))

    # Get a confidence interval based on the quantile of the chi-squared
    # distribution
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression().fit(np.hstack([parsed_mars.koopman_model.transform(input_states)[:, 0], actions[:, 0]]), res_flat)
    print("R² of residual vs (z,u):", lr.score(np.hstack([parsed_mars.koopman_model.transform(input_states)[:, 0], actions[:, 0]]), res_flat))
       
    conf = np.sqrt(scipy.stats.chi2.ppf(
        0.9, output_states[:, 0, :].shape[1]))
    err = diff + conf
    print("Computed error:", err, "(", diff, conf, ")")
    err = err * np.zeros(output_states[:, 0, :].shape[1])
    error = np.minimum(upper_bound, quantile)
    
    
    parsed_mars.error =  np.concatenate((error[:input_states.shape[-1]],  np.zeros(output_states[:, 0, :].shape[1] - input_states.shape[-1])), axis=0)
    # parsed_mars.error =  error



    print("Final Error", parsed_mars.error, error)

    return EnvModel(parsed_mars, domain.lower.detach().numpy(), domain.upper.detach().numpy()), ev_score, r2
