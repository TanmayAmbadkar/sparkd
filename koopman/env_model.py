from typing import Optional, List, Tuple
import numpy as np
import torch
from sklearn.metrics import explained_variance_score, r2_score
from koopman.network import KoopmanLightning, fit_koopman # Assuming these are in a separate file
import seaborn as sns
import matplotlib.pyplot as plt
plt.ioff()


class KoopmanLinearModel:
    """
    Wraps a trained KoopmanLightning model to provide a clean interface for
    accessing the linearized dynamics (A, B, c) and the computed error bound (eps).
    """
    def __init__(self, koopman_model: KoopmanLightning, original_s_dim: int):
        self.koopman_model = koopman_model
        self.s_dim = koopman_model.hparams.state_dim + koopman_model.hparams.embed_dim
        self.original_s_dim = original_s_dim
        self.error_bound = np.zeros(self.s_dim)

    def predict_trajectory(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """
        Predicts a multi-step trajectory in the latent space.

        Args:
            states (np.ndarray): Shape (N, H+1, s_dim) or (H+1, s_dim).
            actions (np.ndarray): Shape (N, H, a_dim) or (H, a_dim).

        Returns:
            np.ndarray: The predicted latent trajectory, shape (N, H, latent_dim).
        """
        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.koopman_model.device)
        actions_tensor = torch.tensor(actions, dtype=torch.float32, device=self.koopman_model.device)

        if states_tensor.dim() == 2:
            states_tensor = states_tensor.unsqueeze(0)
        if actions_tensor.dim() == 2:
            actions_tensor = actions_tensor.unsqueeze(0)
            
        with torch.no_grad():
            pred_latents = self.koopman_model.forward(states_tensor, actions_tensor)

        return pred_latents.cpu().numpy()

    def get_linear_dynamics(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts the learned A, B, and c matrices from the Koopman model.
        """
        # The model is fixed, so we don't need a specific point.
        with torch.no_grad():
            A, B, c = self.koopman_model.transition()
        
        A_np = A.cpu().numpy()
        B_np = B.cpu().numpy()
        c_np = c.cpu().numpy()
        
        return A_np, B_np, c_np

    def get_matrix_at_point(self, point: np.ndarray, s_dim: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the pre-computed linear model and error bound.
        """
        A, B, c = self.get_linear_dynamics()
        M = np.hstack((A, B, c[:, None]))
        return M, self.error_bound

    def __str__(self):
        return "KoopmanLinearModel"


def get_environment_model(
    input_states: np.ndarray,
    actions: np.ndarray,
    output_states: np.ndarray,
    latent_dim: int = 4,
    horizon: int = 5,
    epochs: int = 50,
    percentile: int = 99,
    koopman_model: Optional[KoopmanLightning] = None,
) -> Tuple[KoopmanLinearModel, dict, np.ndarray, np.ndarray]:
    """
    Trains a Koopman dynamics model and computes its error bound.

    Args:
        input_states (np.ndarray): Trajectories of input states.
        actions (np.ndarray): Trajectories of actions.
        output_states (np.ndarray): Trajectories of ground-truth next states.
        latent_dim (int): The number of learned latent dimensions.
        horizon (int): The prediction horizon for training.
        epochs (int): Number of training epochs.
        koopman_model (Optional[KoopmanLightning]): An existing model to continue training.

    Returns:
        A tuple containing:
        - KoopmanLinearModel: The wrapped, trained model.
        - dict: A dictionary of one-step performance metrics (EV, R2).
        - np.ndarray: The mean used for state normalization.
        - np.ndarray: The standard deviation used for state normalization.
    """
    # 1. Normalize Data
    state_shape = input_states.shape[-1]
    means = np.mean(input_states.reshape(-1, state_shape), axis=0)
    stds = np.std(input_states.reshape(-1, state_shape), axis=0)
    stds[stds < 1e-6] = 1.0  # Avoid division by zero
    
    
    input_states_norm = (input_states - means) / stds
    output_states_norm = (output_states - means) / stds

    # 2. Initialize or Continue Training
    if koopman_model is None:
        koopman_model = KoopmanLightning(state_shape, latent_dim, actions.shape[-1], horizon)
    
    fit_koopman(input_states_norm, actions, output_states_norm, koopman_model, horizon, epochs=epochs)

    # 3. Wrap the model in our clean interface
    linear_model = KoopmanLinearModel(koopman_model, original_s_dim=state_shape)

    # 4. Evaluate and Compute Error Bound
    # Predict the full trajectory to get multi-step predictions

    # Pick a random idx set for validation, where size of validation is min(100000, size of dataset)
    
    idx = np.random.choice(input_states.shape[0], size=min(100000, input_states.shape[0]), replace=False)
    pred_latents_traj = linear_model.predict_trajectory(
        input_states_norm[idx], 
        actions[idx]
    )

    # Get ground-truth latent states for the entire trajectory
    with torch.no_grad():
        flat_true_states = torch.tensor(output_states_norm[idx], dtype=torch.float32)
        flat_true_latents = koopman_model.embedding_net(flat_true_states.view(-1, state_shape))
        true_latents_traj = flat_true_latents.view(*pred_latents_traj.shape).numpy()
    
    print("\n--- Step-wise Model Accuracy Evaluation ---")
    final_metrics = {}
    for i in range(horizon):
        # Predicted latents at step i (0-indexed)
        pred_latents_step = pred_latents_traj[:, i, :]
        # Ground truth latents are at step i+1
        true_latents_step = true_latents_traj[:, i, :]

        # For interpretable metrics, compare only the original state dimensions
        pred_states_step = pred_latents_step[:, :state_shape]
        true_states_step = true_latents_step[:, :state_shape]

        mse = np.mean((pred_states_step - true_states_step)**2)
        ev_score = explained_variance_score(true_states_step.flatten(), pred_states_step.flatten())
        r2 = r2_score(true_states_step.flatten(), pred_states_step.flatten())

        print(f"Horizon {i}: MSE={mse:.6f}, EV={ev_score:.4f}, R2={r2:.4f}")

        # Store the metrics for the first step to return
        if i == 0:
            final_metrics['explained_variance'] = ev_score
            final_metrics['r2_score'] = r2

    # Calculate robust error bound based on one-step prediction residuals
    one_step_residuals = np.abs(pred_latents_traj[:, 0, :] - true_latents_traj[:, 0, :])
    error_bound = np.percentile(one_step_residuals, percentile, axis=0)
    linear_model.error_bound = error_bound
    
    print(f"\nComputed Error Bound (eps) using 99th percentile of 1-step error: {error_bound}")
    
    # --- ADDED: Code to plot the boxplot of residuals ---
    # plt.figure(figsize=(12, 6))
    
    # sns.boxplot(data=one_step_residuals)
    # plt.title('Boxplot of One-Step Prediction Residuals per Latent Dimension')
    # plt.xlabel('Latent Dimension Index')
    # plt.ylabel('Absolute Error')
    # plt.grid(True)
    # plt.savefig("residual_boxplot.png")
    # plt.close()
    # print("Saved residual boxplot to residual_boxplot.png")

    return linear_model, final_metrics['explained_variance'], final_metrics['r2_score'], means, stds

