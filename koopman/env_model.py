"""
environment_model.py

This module implements the pipeline for learning a Koopman Operator-based
linear environment model (SPARKD) and estimating its error bounds.

Features:
- Global Linearization using Koopman Operator.
- Adaptive Error: K-Means (Orientation & Magnitude) based on 1-step residual.
- Geometric Optimization: PCA-Rotated bounding boxes (toggleable).
- OOD Detection: Distance-based check using Cluster Radii.
- Automated tuning of cluster count (K) using the Elbow Method.
- Diagnostic Tools: Max Abs Signed Error check (1-Step Only).
"""

from typing import Optional, Tuple, List, Union
import numpy as np
import torch
from sklearn.metrics import explained_variance_score, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors 
import matplotlib.pyplot as plt
# Assuming KoopmanLightning and fit_koopman are defined in koopman.network
from koopman.network import KoopmanLightning, fit_koopman 

class KoopmanLinearModel:
    """
    Wraps a learned Koopman model.
    Uses K-Means to determine Error Orientation (V) and Magnitude (eps) 
    based on the 1-step residual. Error is then analytically propagated in the shield.
    """

    def __init__(self, koopman_model: KoopmanLightning, original_s_dim: int, 
                 adaptive_error: bool = False, use_pca: bool = True,
                 ood_thresh_factor: float = 1.0):
        self.koopman_model = koopman_model
        # Latent dim = State dim + Embedding dim
        self.s_dim = koopman_model.hparams.state_dim + koopman_model.hparams.embed_dim
        self.original_s_dim = original_s_dim
        self.device = koopman_model.device
        self.horizon = koopman_model.hparams.horizon

        # Configuration
        self.adaptive_error = adaptive_error
        self.use_pca = use_pca
        self.ood_thresh_factor = ood_thresh_factor
        
        # Fallbacks: Global bound is now a single vector (D,)
        self.global_bound = np.zeros(self.s_dim)

        # Adaptive Model Components (Single list of size K)
        self.kmeans_model: Optional[KMeans] = None
        self.cluster_rotations: Optional[List[np.ndarray]] = None # List of size K
        self.cluster_bounds: Optional[List[np.ndarray]] = None     # List of size K
        self.cluster_radii: Optional[np.ndarray] = None 
        
        # Diagnostics
        self.stored_residuals: Optional[np.ndarray] = None # Shape (N, D)
        self.stored_z: Optional[np.ndarray] = None


    def set_adaptive_model(self, kmeans: KMeans, z_train: np.ndarray, 
                        residuals: np.ndarray, percentile: float = 99.0,
                        variance_threshold: float = 1.0):
        """
        Trains the adaptive error components using 1-step residuals.
        Calculates the Hybrid Error Bound: Max Observed Error + Statistical Buffer (3-sigma).
        residuals shape: (N_samples, s_dim)
        """
        self.kmeans_model = kmeans
        n_clusters = kmeans.n_clusters
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        STATISTICAL_CONFIDENCE_FACTOR = 3.0 # Simple 3-sigma rule

        # Store for diagnostics
        self.stored_residuals = residuals
        self.stored_z = z_train

        # Initialize storage (Lists of size K)
        self.cluster_rotations = []
        self.cluster_bounds = []
        self.cluster_radii = np.zeros(n_clusters)
        
        print(f"Computing 1-Step Cluster Stats (K={n_clusters}, PCA={self.use_pca})...")
        
        for k in range(n_clusters):
            mask = (labels == k)
            
            # --- 1. CLUSTER RADIUS (Based on Initial State z_0) ---
            if np.sum(mask) > 0:
                cluster_points = z_train[mask]
                dists = np.linalg.norm(cluster_points - centroids[k], axis=1)
                self.cluster_radii[k] = np.max(dists)
            else:
                self.cluster_radii[k] = 0.0

            # --- 2. 1-STEP ERROR BOUNDS ---
            if np.sum(mask) < 10:
                # Fallback for tiny clusters
                self.cluster_rotations.append(np.eye(self.s_dim))
                self.cluster_bounds.append(self.global_bound)
                continue

            # Shape: (N_in_cluster, s_dim)
            err_k = residuals[mask]
            
            # --- ADAPTIVE BOUND CALCULATION ---
            if self.use_pca:
                # A. PCA (Oriented Bounding Box - OBB)
                pca = PCA(n_components=None)
                pca.fit(err_k)
                V = pca.components_.T  # Rotation Matrix
                
                # B. Rotate residuals
                residuals_aligned = err_k @ V 
            else:
                # NO PCA (Axis-Aligned Bounding Box - AABB)
                V = np.eye(self.s_dim) # Identity rotation
                residuals_aligned = err_k # Residuals are already aligned with original axes

            # --- HYBRID BOUND CALCULATION ---
            
            # 1. Empirical Worst-Case (diff): Max magnitude along the aligned axes
            # (This replaces the percentile calculation)
            diff = np.percentile(np.abs(residuals_aligned), percentile, axis=0)

            # 2. Statistical Confidence Term (conf): Based on error standard deviation
            # Note: np.std computes std over the columns (axes), using axis=0
            sigma_error_aligned = np.std(residuals_aligned, axis=0)
            
            # 3. Apply Statistical Factor (e.g., 3-sigma or Chi-squared equivalent)
            # We use a simple 3-sigma rule here, but you can swap in the Chi-squared term if scipy is available
            conf = sigma_error_aligned * STATISTICAL_CONFIDENCE_FACTOR 
            
            # 4. Hybrid Bound: Max Observed + Statistical Buffer
            eps_hybrid = diff + conf 
            
            # --- Spectral Pruning (Still applies to OBB/PCA bounds) ---
            if self.use_pca:
                # Use PCA explained variance for pruning
                expl_var = np.cumsum(pca.explained_variance_ratio_)
                n_keep = np.searchsorted(expl_var, variance_threshold) + 1
                if n_keep < self.s_dim:
                    eps_hybrid[n_keep:] = 1e-6
            
            # Final update
            self.cluster_rotations.append(V)
            self.cluster_bounds.append(eps_hybrid)
    
    def predict_trajectory(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.float32, device=self.device)

        if states_tensor.dim() == 2:
            states_tensor = states_tensor.unsqueeze(0)
        if actions_tensor.dim() == 2:
            actions_tensor = actions_tensor.unsqueeze(0)

        with torch.no_grad():
            pred_latents = self.koopman_model.forward(states_tensor, actions_tensor)

        return pred_latents.cpu().numpy()

    def get_linear_dynamics(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with torch.no_grad():
            A, B, c = self.koopman_model.transition()
        return A.cpu().numpy(), B.cpu().numpy(), c.cpu().numpy()

    def get_matrix_at_point(self, point: np.ndarray, s_dim: int, **kwargs) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Returns:
            M: Linear Dynamics
            adaptive_info: A single tuple (V_0, eps_0)
        """
        # 1. Get Linear Dynamics
        with torch.no_grad():
            A, B, c = self.koopman_model.transition()
        M = np.hstack((A.cpu().numpy(), B.cpu().numpy(), c.cpu().numpy()[:, None]))
        
        # Default Fallback (Global Bound)
        fallback_info = (np.eye(self.s_dim), self.global_bound)

        if not self.adaptive_error or self.kmeans_model is None:
            return M, fallback_info

        # 2. Lift State
        with torch.no_grad():
            # point contains both state and action, only state is passed to embedding_net
            x_tensor = torch.tensor(point, dtype=torch.float32, device=self.device).unsqueeze(0)
            z_tensor = self.koopman_model.embedding_net(x_tensor[:, :self.koopman_model.hparams.state_dim])
            z_np = z_tensor.cpu().numpy() 

        # 3. K-Means Lookup
        dists_to_centroids = self.kmeans_model.transform(z_np)[0]
        cluster_idx = np.argmin(dists_to_centroids)
        dist_to_center = dists_to_centroids[cluster_idx]
        
        # --- OOD CHECK ---
        max_radius = self.cluster_radii[cluster_idx]
        if dist_to_center > (max_radius * self.ood_thresh_factor):
            return M, fallback_info

        # 4. Return Single Adaptive Bound
        V = self.cluster_rotations[cluster_idx]
        eps = self.cluster_bounds[cluster_idx]
        
        return M, (V, eps)

    def __str__(self):
        mode = "Adaptive+PCA" if self.adaptive_error and self.use_pca else ("Adaptive+AABB" if self.adaptive_error else "Fixed")
        return f"KoopmanLinearModel(mode={mode})"


def compute_pca_bounds_for_cluster(
    residuals: np.ndarray, 
    percentile: float, 
    variance_threshold: float = 1.0,
    use_pca: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes PCA rotation and bounds (Used for Auto-Tuning).
    """
    dim = residuals.shape[1]
    if residuals.shape[0] < 2:
        return np.eye(dim), np.percentile(np.abs(residuals), percentile, axis=0)

    if use_pca:
        # Standard PCA calculation (OBB)
        covariance = np.cov(residuals, rowvar=False)
        eigenvalues, V = np.linalg.eigh(covariance)
        residuals_rot = residuals @ V
        eps_pca = np.percentile(np.abs(residuals_rot), percentile, axis=0)
        
        # Spectral Pruning logic
        total_var = np.sum(eigenvalues)
        if total_var > 1e-9:
            sorted_evals_desc = np.flip(eigenvalues)
            cumsum_var = np.cumsum(sorted_evals_desc)
            n_top = np.searchsorted(cumsum_var, total_var * variance_threshold) + 1
            n_tail = dim - n_top
            if n_tail > 0:
                eps_pca[n_tail:] = 1e-6
    else:
        # No PCA calculation (AABB)
        V = np.eye(dim)
        eps_pca = np.percentile(np.abs(residuals), percentile, axis=0)
            
    return V, eps_pca


def auto_tune_clusters(
    z_data: np.ndarray,
    residuals: np.ndarray, # Shape (N, D) - 1-step residuals
    percentile: float = 99.0,
    min_samples_per_cluster: int = 1000,
    variance_threshold: float = 1.0,
    use_pca: bool = True
) -> Tuple[KMeans, np.ndarray, List[np.ndarray], dict]:
    """
    Sweeps K and selects the optimal cluster count using the Elbow Method (Kneedle).
    """
    residuals_metric = residuals 
    
    N = z_data.shape[0]
    max_k = int(N / min_samples_per_cluster)
    raw_candidates = [10, 50, 100, 200, 500, 1000] + [max_k]
    candidates = sorted(list(set([k for k in raw_candidates if 1 < k <= max_k])))

    if not candidates:
        candidates = [max(1, int(N / 1000))]

    print(f"  Auto-tuning clusters (Elbow Method). Candidates: {candidates}")
    results_history = [] 

    for k in candidates:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=3).fit(z_data)
        labels = kmeans.labels_
        current_pca_bounds = np.zeros((k, residuals_metric.shape[1]))
        current_rotations = []

        for cluster_idx in range(k):
            mask = (labels == cluster_idx)
            if np.sum(mask) > 10:
                V, eps = compute_pca_bounds_for_cluster(
                    residuals_metric[mask], percentile, variance_threshold, use_pca=use_pca
                )
            else:
                # Fallback for small clusters
                V = np.eye(residuals_metric.shape[1])
                eps = np.percentile(np.abs(residuals_metric), percentile, axis=0)
            
            eps = np.maximum(eps, 1e-9)
            current_pca_bounds[cluster_idx] = eps
            current_rotations.append(V)

        cluster_log_vols = np.sum(np.log(current_pca_bounds), axis=1)
        avg_log_vol = np.mean(cluster_log_vols)

        results_history.append({
            'k': k, 'score': avg_log_vol, 'kmeans': kmeans
        })
        print(f"    K={k}: Avg Log-Vol={avg_log_vol:.4f}")

    if len(results_history) < 3:
        best_result = min(results_history, key=lambda x: x['score'])
    else:
        ks = np.array([r['k'] for r in results_history])
        scores = np.array([r['score'] for r in results_history])
        ks_norm = (ks - ks.min()) / (ks.max() - ks.min())
        scores_norm = (scores.max() - scores) / (scores.max() - scores.min())
        
        start_point = np.array([ks_norm[0], scores_norm[0]])
        end_point = np.array([ks_norm[-1], scores_norm[-1]])
        line_vec = end_point - start_point
        distances = []
        for i in range(len(ks)):
            point = np.array([ks_norm[i], scores_norm[i]])
            vec_from_start = point - start_point
            cross_prod = line_vec[0] * vec_from_start[1] - line_vec[1] * vec_from_start[0]
            dist = np.abs(cross_prod) / np.linalg.norm(line_vec)
            distances.append(dist)
        best_idx = np.argmax(distances)
        best_result = results_history[best_idx]
        print(f"  Elbow Detected at K={best_result['k']} (Score: {best_result['score']:.4f})")

    # Return None for bounds/rotations as they are re-computed properly in set_adaptive_model
    return best_result['kmeans'], None, None, {'avg_log_vol': best_result['score']}


def get_environment_model(
    input_states: np.ndarray,
    actions: np.ndarray,
    output_states: np.ndarray,
    latent_dim: int = 4,
    horizon: int = 5,
    epochs: int = 50,
    percentile: int = 99,
    koopman_model: Optional[KoopmanLightning] = None,
    adaptive_error: bool = False,
    use_pca: bool = False,
    n_clusters: Optional[int] = None,
    variance_threshold: float = 1.0,
) -> Tuple[KoopmanLinearModel, float, float, np.ndarray, np.ndarray]:

    # 1. Normalize Data
    state_shape = input_states.shape[-1]
    means = np.mean(input_states.reshape(-1, state_shape), axis=0)
    stds = np.std(input_states.reshape(-1, state_shape), axis=0)
    stds[stds < 1e-6] = 1.0

    input_states_norm = (input_states - means) / stds
    output_states_norm = (output_states - means) / stds

    # 2. Train Koopman
    if koopman_model is None:
        koopman_model = KoopmanLightning(state_shape, latent_dim, actions.shape[-1], horizon)
    
    fit_koopman(input_states_norm, actions, output_states_norm, koopman_model, horizon, epochs=epochs)

    linear_model = KoopmanLinearModel(
        koopman_model, original_s_dim=state_shape, adaptive_error=adaptive_error, 
        use_pca=use_pca
    )

    # 3. Generate Predictions
    print("\n--- Generating predictions... ---")
    pred_latents_traj = linear_model.predict_trajectory(input_states_norm, actions)

    with torch.no_grad():
        flat_true_states = torch.tensor(output_states_norm, dtype=torch.float32, device=linear_model.device)
        flat_true_z = koopman_model.embedding_net(flat_true_states.view(-1, state_shape))
        true_latents_traj = flat_true_z.view(*pred_latents_traj.shape).cpu().numpy()

    # 4. Metrics
    print("\n--- Step-wise Model Accuracy Evaluation ---")
    final_metrics = {}
    eval_horizon = min(horizon, pred_latents_traj.shape[1])
    for i in range(eval_horizon):
        pred_x = pred_latents_traj[:, i, :state_shape]
        true_x = true_latents_traj[:, i, :state_shape]
        
        ev_score = explained_variance_score(true_x.flatten(), pred_x.flatten())
        r2 = r2_score(true_x.flatten(), pred_x.flatten())
        print(f"Horizon {i}: EV={ev_score:.4f}, R2={r2:.4f}")
        if i == 0:
            final_metrics['ev'] = ev_score
            final_metrics['r2'] = r2

    # 5. Error Calculation (1-STEP ONLY)
    # Shape: (N, D)
    signed_residuals = true_latents_traj[:, 0, :] - pred_latents_traj[:, 0, :]
    
    # Global Bound is now a single vector (D,)
    linear_model.global_bound = np.percentile(np.abs(signed_residuals), percentile, axis=0)
    print(f"\nComputed Global Error Bound: {linear_model.global_bound}...")

    # 6. TRAIN ADAPTIVE ERROR MODEL
    if adaptive_error:
        print(f"\nTraining Adaptive Error Model (1-Step)...")
        with torch.no_grad():
            input_x = torch.tensor(input_states_norm[:, 0, :], dtype=torch.float32, device=linear_model.device)
            z_train = koopman_model.embedding_net(input_x).cpu().numpy()

        if n_clusters is not None:
            print(f"  Manual K={n_clusters}")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(z_train)
            linear_model.set_adaptive_model(
                kmeans, z_train, signed_residuals, 
                percentile=float(percentile), variance_threshold=variance_threshold
            )
        else:
            best_km, _, _, _ = auto_tune_clusters(
                z_train, signed_residuals, float(percentile), 
                variance_threshold=variance_threshold, use_pca=use_pca
            )
            linear_model.set_adaptive_model(
                best_km, z_train, signed_residuals, 
                percentile=float(percentile), variance_threshold=variance_threshold
            )

        print("Adaptive Error Model trained successfully.")

    # 7. DIAGNOSTIC PLOTTING (1-Step Only)
    try:
        dataset_size = input_states.shape[0]
        state_shape = input_states.shape[-1]
        
        # We only plot the single step (h=0) data
        step_res = signed_residuals 
        step_bound = linear_model.global_bound 

        plt.figure(figsize=(10, 6))
        plt.boxplot(step_res)
        
        # Overlay Global Bound
        feature_indices = np.arange(1, step_res.shape[1] + 1)
        plt.plot(feature_indices, step_bound, 'r*', label=f'{percentile}th % Bound', markersize=10)
        plt.plot(feature_indices, -step_bound, 'r*', markersize=10)

        # CHECK CLUSTER CENTERS (Max Abs Signed Residual)
        if adaptive_error and linear_model.kmeans_model is not None:
            print("\n--- Checking Cluster Center Errors (Max Abs Signed) ---")
            kmeans = linear_model.kmeans_model
            z_train = linear_model.stored_z
            residuals = linear_model.stored_residuals # 1-step residuals
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_
            
            # Center residuals will store the worst-case signed error for each cluster
            center_residuals = np.zeros((len(centers), residuals.shape[1]))
            
            for k in range(len(centers)):
                mask = (labels == k)
                if np.sum(mask) == 0:
                    continue
                    
                cluster_res = residuals[mask]
                abs_cluster_res = np.abs(cluster_res)
                max_indices = np.argmax(abs_cluster_res, axis=0)
                
                # Construct the worst-case signed vector
                for d in range(residuals.shape[1]):
                    center_residuals[k, d] = cluster_res[max_indices[d], d]
            
            # Check bounds on raw center errors
            in_bound = np.abs(center_residuals) <= step_bound
            safe_counts = np.sum(np.all(in_bound, axis=1))
            print(f"  {safe_counts}/{len(centers)} Cluster Centers have residuals within Global Bound.")
            
            for f in range(state_shape):
                x_vals = np.full(len(centers), f + 1)
                y_vals = center_residuals[:, f]
                if f == 0:
                    plt.plot(x_vals, y_vals, 'bo', alpha=0.6, markersize=4, label='Centers (Max Abs Signed)')
                else:
                    plt.plot(x_vals, y_vals, 'bo', alpha=0.6, markersize=4)

        plt.legend()
        plt.title(f'Step 1 Residual Distribution (N={dataset_size})')
        plt.xlabel('Feature Index')
        plt.ylabel('Error')
        plt.grid(True, linestyle='--', alpha=0.7)
        plot_filename = f"residual_boxplot_{dataset_size}.png"
        plt.savefig(plot_filename)
        plt.close()
        print(f"Saved residual boxplot to {plot_filename}")

    except Exception as e:
        print(f"Warning: Failed to save residual boxplot: {e}")

    return linear_model, final_metrics['ev'], final_metrics['r2'], means, stds