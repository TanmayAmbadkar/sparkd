from typing import Optional, List, Callable
import numpy as np
import scipy.stats
from e2c.e2c_model import E2CPredictor, fit_e2c
from abstract_interpretation.verification import get_constraints, get_ae_bounds, get_variational_bounds
from abstract_interpretation import domains
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import explained_variance_score
from typing import Union



class MarsE2cModel:
    """
    A model that uses the E2CPredictor to obtain A, B, and c matrices
    and provides a similar interface to MARSModel.
    """
    def __init__(self, e2c_predictor: E2CPredictor, s_dim=None, original_s_dim = None):
        self.e2c_predictor = e2c_predictor
        self.s_dim = s_dim
        self.original_s_dim = original_s_dim

    def __call__(self, point,  normalized: bool = False) -> np.ndarray:
        """
        Predict the next state given the current state x and action u.
        """
        
        x_norm = point[:self.original_s_dim ]
        u_norm = point[self.original_s_dim:]
        # Convert to tensors
        x_tensor = torch.tensor(x_norm, ).unsqueeze(0)
        u_tensor = torch.tensor(u_norm, ).unsqueeze(0)

        # Use E2CPredictor to predict next state
        z_t_next = self.e2c_predictor.get_next_state(x_tensor, u_tensor)

        # Predict next latent state
        

        return z_t_next

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

        # 2. Split into state (x_norm) and action (u_norm)
        x_norm = point[:s_dim]
        u_norm = point[s_dim:]

        # 3. Convert to torch tensors
        x_tensor = torch.tensor(x_norm, ).unsqueeze(0)
        u_tensor = torch.tensor(u_norm, ).unsqueeze(0)

        # 4. Run the E2C transition:
        #    Returns (z_next, z_next_mean, A_t, B_t, c_t, v_t, r_t)

        with torch.no_grad():
            z_next, z_next_mean, A_t, B_t, c_t, v_t, r_t = self.e2c_predictor.transition(
                x_tensor, x_tensor, u_tensor
            )

        # 5. Convert PyTorch tensors to NumPy, remove batch dimension
        A_t = A_t.detach().cpu().numpy().squeeze(0)    # shape [s_dim, s_dim]
        B_t = B_t.detach().cpu().numpy().squeeze(0)    # shape [s_dim, u_dim]
        c_t = c_t.detach().cpu().numpy().squeeze(0)    # shape [s_dim]

        # 6. Construct M by stacking [A | B | c], giving shape [s_dim, s_dim + u_dim + 1]
        #    Note: c_t[:, None] is the bias column
        M = np.hstack((A_t, B_t, c_t[:, None]))

        # 7. Compute eps as the diagonal of A_t @ A_t^T.
        #    That yields a 1D array of length s_dim.
        A_tA_tT = A_t @ A_t.T  # shape [s_dim, s_dim]
        # eps = np.diag(A_tA_tT) # shape [s_dim]
        eps = np.zeros_like(c_t)

        return M, eps



    def __str__(self):
        return "MarsE2cModel using E2CPredictor"

class RewardModel:
    
    def __init__(
            self, 
            input_size:int, 
            input_mean: np.ndarray,
            input_std: np.ndarray,
            rew_mean: np.ndarray,
            rew_std: np.ndarray\
        ):
        
        self.model = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        
        self.input_mean = input_mean
        self.input_std = input_std
        self.rew_mean = rew_mean
        self.rew_std = rew_std
        
    def train(self, X, y):
            
        # Convert inputs and rewards to tensors
        X = torch.Tensor(X)
        rewards = torch.Tensor(y)
        
        # Define the loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0003)
        
        # Create DataLoader for batching
        dataset = TensorDataset(X, rewards)
        dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
        
        # Training loop
        epochs = 0
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            all_preds = []
            all_targets = []
            for batch_X, batch_rewards in dataloader:
                # Zero the gradients
                optimizer.zero_grad()
                
                # Forward pass
                predictions = self.model(batch_X)
                
                # Compute loss
                loss = criterion(predictions.squeeze(), batch_rewards)
                
                # Backward pass and update weights
                loss.backward()
                optimizer.step()
                
                # Accumulate loss
                total_loss += loss.item()
                
                # Save predictions and targets for EV computation
                all_preds.append(predictions.squeeze().detach().cpu().numpy())
                all_targets.append(batch_rewards.detach().cpu().numpy())
            
            # Calculate average loss over epoch
            avg_loss = total_loss / len(dataloader)
            
            # Concatenate predictions and targets
            all_preds = np.concatenate(all_preds)
            all_targets = np.concatenate(all_targets)
            
            # Compute explained variance:
            ev = 1 - np.var(all_targets - all_preds) / np.var(all_targets)
            
            # Print loss and explained variance
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Explained Variance: {ev:.4f}")

        
    def __call__(self, X):
        
        X = (X - self.input_mean)/(self.input_std)
        with torch.no_grad():
            rew = self.model(torch.Tensor(X).reshape(1, -1))
        
        return rew.detach().numpy().reshape(-1, ) * (self.rew_std) + self.rew_mean

        

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
            symb_reward: RewardModel,
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
        self.symb_reward = symb_reward
        self
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
        
        rew = self.symb_reward(np.concatenate((inp, symb[0]), axis = 0))[0]
            
        return np.clip(symb[0], self.observation_space_low, self.observation_space_high), rew

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
        e2c_predictor = None,
        epochs: int = 50) -> EnvModel:

    
    means = np.mean(input_states, axis=0)
    stds = np.std(input_states, axis=0)
    stds[np.equal(np.round(stds, 1), np.zeros(*stds.shape))] = 1

    print("Means:", means)
    print("Stds:", stds)
    
    # means = np.zeros_like(means)
    # stds = np.ones_like(stds)
    
    # if e2c_predictor is not None:
    #     means = e2c_predictor.mean + 0.001 * (means - e2c_predictor.mean)
    #     stds = e2c_predictor.std + 0.001 * (means - e2c_predictor.mean)
    
    input_states = (input_states - means) / stds
    output_states = (output_states - means) / stds

    
    # domain.lower = (domain.lower - means) / stds
    # domain.upper = (domain.upper - means) / stds
    # print("Input states:", input_states)
    
    if e2c_predictor is None:
        e2c_predictor = E2CPredictor(input_states.shape[-1], latent_dim, actions.shape[-1], horizon = horizon)
    fit_e2c(input_states, actions, output_states, e2c_predictor, e2c_predictor.horizon, epochs=epochs)

    e2c_predictor.mean = means
    e2c_predictor.std = stds

    
    # lows, highs = get_ae_bounds(e2c_predictor, domain)
    
    # lows = lows.detach().numpy()
    # highs = highs.detach().numpy()
    
    input_states= input_states.reshape(-1, input_states.shape[-1])
    actions = actions.reshape(-1, actions.shape[-1])
    output_states = output_states.reshape(-1, output_states.shape[-1])
    rewards = rewards.reshape(-1, 1)
    # input_states = e2c_predictor.transform(input_states)
    # output_states = e2c_predictor.transform(output_states)
    
    actions_min = actions.min(axis=0)
    actions_max = actions.max(axis=0)
    rewards_min = rewards.min()
    rewards_max = rewards.max()

    print("State stats:", input_states.min(axis = 0), input_states.max(axis = 0))
    print("Action stats:", actions_min, actions_max)
    print("Reward stats:", rewards_min, rewards_max)
    
    
    parsed_mars = MarsE2cModel(e2c_predictor, latent_dim, input_states.shape[-1])
    
    X = np.concatenate((input_states * stds + means, actions), axis=1)
    Yh = np.array([parsed_mars(state, normalized=True) for state in X]).reshape(input_states.shape[0], -1)
    
    print("Model estimation error:", np.mean((Yh - (output_states* stds + means))**2))
    print("Explained Variance Score:", explained_variance_score(
        output_states * stds + means, Yh))

    print(Yh[:10])
    print((output_states* stds + means )[:10])
    
    
    # Get the maximum distance between a predction and a datapoint
    diff = np.amax(np.abs(Yh - (output_states* stds + means)))

    # Get a confidence interval based on the quantile of the chi-squared
    # distribution
    conf = data_stddev * np.sqrt(scipy.stats.chi2.ppf(
        0.9, output_states.shape[1]))
    err = diff + conf
    print("Computed error:", err, "(", diff, conf, ")")
    parsed_mars.error = err

    rew_mean, rew_std = np.mean(rewards), np.std(rewards)
    
    print("Input mean:", means)
    print("Input std:", stds)
    
    rewards = (rewards - rew_mean) / (rew_std)

    # Now, instead of using only output_states as input to the reward model,
    # we create a new input X_rew by concatenating (state, action, next_state)
    X_rew = np.concatenate((input_states, actions, output_states), axis=1)
    
    # Create a reward model with input size equal to the new concatenated dimension.
    input_mean = np.concatenate((means, np.zeros(actions.shape[1]), means))
    input_std= np.concatenate((stds, np.ones(actions.shape[1]), stds))
    parsed_rew = RewardModel(X_rew.shape[1], input_mean, input_std, rew_mean, rew_std)
    parsed_rew.train(X_rew, rewards)

    print(parsed_mars)
    print("Model MSE:", np.mean(np.sum((Yh - output_states)**2, axis=1)))
    # print(reward_symb.summary())

    return EnvModel(parsed_mars, parsed_rew, domain.lower.detach().numpy(), domain.upper.detach().numpy())