import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from src.vdk_shield import VDK_Shield, TabularEncoder, TabularDecoder, VisualEncoder, VisualDecoder, stage1_loss, stage2_loss
from tqdm import tqdm

def train_vdk(
    data_path,
    output_path,
    epochs_dyn=50,
    epochs_cbf=20,
    batch_size=256,
    lr=1e-4,
    latent_dim=32,
    gpu=False,
    pretrained_path=None
):
    """
    Train VDK Shield with Spectral Koopman Dynamics.
    """
    # --- 1. Load Data ---
    print(f"Loading data from {data_path}...")
    data = torch.load(data_path)
    
    states = data['states']          # (N, n)
    actions = data['actions']        # (N, m)
    next_states = data['next_states']# (N, n)
    horizon_targets = data['horizon_targets'] # (N, )
    
    # --- 2. Initialize Model ---
    # states shape is (N, *state_dim) OR (N, T, *state_dim)
    # Check for Time dimension
    # If 3 dims (N, T, D), then tabular sequential.
    # If 2 dims (N, D), then tabular flat.
    # If 4 dims (N, C, H, W), image flat.
    # If 5 dims (N, T, C, H, W), image sequential.
    
    shape = states.shape
    has_time_dim = False
    
    if len(shape) == 2: # (N, D)
        is_image = False
        state_dim = shape[1]
    elif len(shape) == 3: # (N, T, D) or (N, C, H)? 
        # Usually C,H are small but T might be small (5). D is 27.
        # Gym images: (C, H, W). 3 dims. So (N, C, H, W).
        # Wait, if input is (N, C, H), len is 3.
        # But here shape includes N.
        # (N, T, D) -> len 3.
        # (N, C, H, W) -> len 4.
        is_image = False
        has_time_dim = True
        state_dim = shape[2]
    elif len(shape) == 4: # (N, C, H, W) - Image Flat
        is_image = True
        state_dim = shape[1:] # (C, H, W)? No, logic below uses obs_shape
    elif len(shape) == 5: # (N, T, C, H, W) - Image Seq
        is_image = True
        has_time_dim = True
        state_dim = shape[2:]
        
    obs_shape = states.shape[1:] # Careful, this includes T if has_time_dim
    
    if is_image:
        print("Detected Image Data. Using VisualEncoder/Decoder and /255 Normalization.")
         # ... (Assuming image logic needs similar update if T present, but focusing on Tabular now)
         # For now, simplistic handling or error if T present for images
        if has_time_dim:
             # Normalize over (N, T)
             print(f"Image Data Stats (Pre-Norm): Min={states.min():.4f}, Max={states.max():.4f}, Mean={states.mean():.4f}")
             
             if states.max() > 1.0:
                 states = states / 255.0
                 next_states = next_states / 255.0
                 print("Applied /255.0 normalization.")
             else:
                 print("Skipped /255.0 normalization (Data already <= 1.0).")

             print(f"Image Data Stats (Post-Norm): Min={states.min():.4f}, Max={states.max():.4f}, Mean={states.mean():.4f}")
             in_channels = shape[2] 

             encoder = VisualEncoder(latent_dim=latent_dim, in_channels=in_channels)
             decoder = VisualDecoder(latent_dim=latent_dim, out_channels=in_channels)
        else:
             if states.max() > 1.0:
                 states = states / 255.0
                 next_states = next_states / 255.0
             in_channels = shape[1]
             encoder = VisualEncoder(latent_dim=latent_dim, in_channels=in_channels)
             decoder = VisualDecoder(latent_dim=latent_dim, out_channels=in_channels)
         
        state_mean = torch.zeros(1) # Placeholder
        state_std = torch.ones(1) * 255.0
        
    else:
        print(f"Detected Tabular Data. Shape: {shape}. T-dim: {has_time_dim}")
        
        if has_time_dim:
            # Normalize over (N*T, D)
            flat_states = states.reshape(-1, state_dim)
            state_mean = flat_states.mean(dim=0)
            state_std = flat_states.std(dim=0) + 1e-6
            
            # Broadcast normalization
            # states: (N, T, D), mean: (D)
            states = (states - state_mean) / state_std
            next_states = (next_states - state_mean) / state_std
        else:
            state_mean = states.mean(dim=0)
            state_std = states.std(dim=0) + 1e-6
            states = (states - state_mean) / state_std
            next_states = (next_states - state_mean) / state_std

        encoder = TabularEncoder(state_dim=state_dim, latent_dim=latent_dim)
        decoder = TabularDecoder(latent_dim=latent_dim, state_dim=state_dim)

    # Re-create dataset with normalized data
    # Input s: (N, T, D)
    # Target s_next: (N, T, D) - technically s[1:] vs next_states...
    # ReplayMemory.sample(horizon=H) returns:
    #   states: x_0 ... x_{H-1}
    #   next:   x_1 ... x_H
    # So aligning s and s_next is perfect.
    
    dataset = TensorDataset(states, actions, next_states, horizon_targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Calculate Horizon T
    # states shape is (N, T, D) or (N, T, C, H, W)
    if has_time_dim:
         T_horizon = states.shape[1]
         control_dim = actions.shape[2]
    else:
         T_horizon = 1
         control_dim = actions.shape[1]
         
    print(f"Dataset Horizon T = {T_horizon}, Control Dim = {control_dim}")

    print(f"Initializing VDK Shield (Control: {control_dim}, Latent: {latent_dim})...")

    model = VDK_Shield(encoder=encoder, control_dim=control_dim, decoder=decoder)
    
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pre-trained weights from {pretrained_path}...")
        pt_data = torch.load(pretrained_path)
        # Handle state dict mismatch if necessary (e.g. legacy to new)
        # Assuming we are starting fresh or compatible
        try:
             model.load_state_dict(pt_data['state_dict'])
        except RuntimeError as e:
             print(f"Warning: Could not load exact state dict (architecture changed?): {e}")

    if gpu:
        if torch.cuda.is_available():
            device_name = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_name = 'mps'
        else:
            device_name = 'cpu'
    else:
        device_name = 'cpu'
        
    device = torch.device(device_name)
    print(f"Using device: {device}")
    model = model.to(device)
    
    # --- 3. Stage 1: Variational Koopman Learning (Dynamics) ---
    print("\n=== Stage 1: Training Dynamics ===")
    optimizer_dyn = optim.Adam(model.parameters(), lr=lr)
    
    # Enable gradients
    for param in model.parameters(): param.requires_grad = True

    # Scheduled Sampling settings
    eps = 1.0 # Start with 100% teacher forcing
    eps_decay = 1.0 / max(1, epochs_dyn // 2) # Linear decay over half epochs
    
    spectral_reg_weight = 0.01  # Weight for |lambda| <= 1 constraint
    
    final_dyn_loss = 0.0
    for epoch in range(epochs_dyn):
        total_loss = 0
        
        # Decay epsilon
        eps = max(0.0, eps - eps_decay)
        
        for s, u, s_next, _ in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs_dyn} (eps={eps:.2f})", leave=False):
            s, u, s_next = s.to(device), u.to(device), s_next.to(device)
            # s: (B, T, D)
            
            optimizer_dyn.zero_grad()
            
            # --- Autoregressive Loop ---
            loss_rec = 0
            loss_lin = 0
            # Encode x_0 -> z_0 (Complex)
            if len(s.shape) == 5:
                 current_obs = s[:, 0, :, :, :]
            else:
                 current_obs = s[:, 0, :] # x_0
            
            mu0_re, mu0_im, logvar0_re, logvar0_im = model.encode(current_obs) # (B, d) each
            
            # Sample z_0
            z_re, z_im = model.reparameterize(mu0_re, mu0_im, logvar0_re, logvar0_im)
            
            # We also need Ground Truth Z sequence for 'lin' loss (latent consistency)
            # Encode ALL frames in BATCH (B*T, D) -> (B*T, 4d)
            if len(s.shape) == 5:
                B, T, C, H, W = s.shape
                # Flatten to encode: (B*T, C, H, W)
                flat_s = s.reshape(B*T, C, H, W)
            else:
                B, T, D = s.shape
                # Flatten to encode
                flat_s = s.reshape(B*T, -1)
            
            # Encoder returns 4-tuple directly now
            mu_flat_re, mu_flat_im, _, _ = model.encoder(flat_s) 
            
            gt_z_seq_re = mu_flat_re.reshape(B, T, -1) # (B, T, d)
            gt_z_seq_im = mu_flat_im.reshape(B, T, -1) # (B, T, d)
            
            # Weighted Loss Parameters
            gamma_loss = 0.5
            step_weights = [gamma_loss**i for i in range(T)]
            total_weight = sum(step_weights)
            
            loss_rec_total = 0
            loss_lin_total = 0

            for t in range(T):
                # 1. Decode current prediction -> x_hat_t
                rec_obs = model.decode(z_re, z_im) # (B, D)
                
                # 2. Reconstruction Loss
                target_obs = s[:, t, :] # Ground Truth x_t
                step_rec_loss = torch.mean((rec_obs - target_obs)**2)
                
                # 3. Linearity/Consistency Loss
                # Compare z_t (pred) with Encoder(x_t) (ground truth)
                # V-Final: Match both Real and Imaginary parts
                gt_z_t_re = gt_z_seq_re[:, t, :]
                gt_z_t_im = gt_z_seq_im[:, t, :]
                
                step_lin_loss = torch.mean((z_re - gt_z_t_re)**2) + torch.mean((z_im - gt_z_t_im)**2)
                
                # Accumulate Weighted
                w_t = step_weights[t]
                loss_rec_total += w_t * step_rec_loss
                loss_lin_total += w_t * step_lin_loss
                
                # 4. Predict Next Step (Dynamics)
                if t < T - 1:
                    action_t = u[:, t, :]
                    
                    # Teacher Forcing Decision
                    use_ground_truth = (torch.rand(1).item() < eps)
                    
                    if use_ground_truth:
                         # Feed GT z_t into dynamics
                         curr_z_re = gt_z_t_re
                         curr_z_im = gt_z_t_im
                    else:
                         # Feed recurrent prediction
                         curr_z_re = z_re
                         curr_z_im = z_im
                         
                    z_re, z_im = model.predict_next(curr_z_re, curr_z_im, action_t)
                    
            # Normalize
            loss_rec = loss_rec_total / total_weight
            loss_lin = loss_lin_total / total_weight
            
            # 5. Spectral Regularization
            # |lambda| = r (from new Polar parameterization)
            # Penalty: sum( relu( |lambda| - 1.0 )^2 )
            # lambda_modulus_sq is r^2.
            # We want to penalize if r > 1.
            # Using sqrt(r^2) - 1.
            
            lam_sq = model.dynamics.lambda_modulus_sq
            lam_abs = torch.sqrt(lam_sq + 1e-8)
            loss_spec = torch.sum(torch.relu(lam_abs - 1.0)**2)
            
            # Total
            lambda_pred = 1.0
            lambda_lin = 1.0
            
            loss_total = lambda_pred * loss_rec + lambda_lin * loss_lin + spectral_reg_weight * loss_spec
            
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_dyn.step()
            
            total_loss += loss_total.item()
            
        final_dyn_loss = total_loss / len(loader)
        
        # Diagnostic: Check for Posterior Collapse (or Trivial Code)
        # Calculate spread of Means across the last batch
        with torch.no_grad():
            # mu0_re is (B, d)
            # Variance across Batch dimension: if 0, all inputs map to same z
            spread_re = mu0_re.var(dim=0).mean().item()
            spread_im = mu0_im.var(dim=0).mean().item()
            mean_spread = spread_re + spread_im
            
        print(f"Epoch {epoch+1}: Loss = {final_dyn_loss:.4f} (Spec: {loss_spec.item():.4f}) | LatentSpread: {mean_spread:.2e}")
        
        if mean_spread < 1e-4:
            print("WARNING: Latent Space Collapse Detected! (Spread < 1e-4)")
        
    # --- 4. Stage 2: Robust Safety Value Iteration (CBF) ---
    print("\n=== Stage 2: Training CBF Head ===")
    
    # Freeze Encoder/Dynamics/Decoder
    for param in model.encoder.parameters(): param.requires_grad = False
    for param in model.dynamics.parameters(): param.requires_grad = False
    if model.decoder:
        for param in model.decoder.parameters(): param.requires_grad = False
    
    # Only train w and beta
    model.w.requires_grad = True
    model.beta.requires_grad = True
    
    optimizer_cbf = optim.Adam([model.w, model.beta], lr=lr) 
    
    final_cbf_loss = 0.0
    for epoch in range(epochs_cbf):
        total_loss = 0
        
        for s, u, _, y_h in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs_cbf}", leave=False):
            # s is (B, T, D). We only need s[:, 0]
            # u is (B, T, m). We need u[:, 0] approx for the first step, 
            # but CBF training logic in old code:
            #   predict_next(z0, u) -> z1 -> V(z1) approx y_h?
            #   Usually V(z) predicts safety of z.
            #   Wait, old code:
            #       mu_t = encode(s)
            #       z_next = predict(mu_t, u)
            #       V = cbf(z_next)
            #       loss(V, y_h)
            #   This implies y_h is target for state AFTER u?
            #   If y_h corresponds to s, then V(encode(s)) should match y_h.
            #   Let's check 'compute_horizon_labels'. It computes label for s_t.
            #   So we should train V(z_t) -> y_t.
            #   Why did old code do predict_next?
            #       Maybe checking if taking u leads to safe state?
            #       "V(z) = w^T z + beta"
            #       If we train V(z_next) against y_h, we are saying:
            #       "The safety of the *result* of action u matches y_h".
            #       But y_h is usually computed from s_t (and future).
            #       If y_h is computed at t, it is the safety of having arrived at t.
            #       Let's assume we train V(z_t) directly.
            
            s0 = s[:, 0, :].to(device)
            y_h = y_h.to(device).unsqueeze(1) # (B, 1) target for s0
             
            # Encode (No Grads)
            # Encode (No Grads)
            with torch.no_grad():
                mu_re, mu_im, _, _ = model.encode(s0)
                # Use mean for CBF evaluation
                z_re = mu_re
                z_im = mu_im
                
            # CBF Score on CURRENT latent z (not next)
            # This is more robust for "Safety of state s"
            V = model.cbf_score(z_re, z_im)
            
            # Loss
            loss, _ = stage2_loss(V, y_h, w=model.w)
            
            optimizer_cbf.zero_grad()
            loss.backward()
            optimizer_cbf.step()
            
            total_loss += loss.item()
            
        final_cbf_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}: Loss = {final_cbf_loss:.4f}")
        
    # --- 5. Return/Save Model ---
    save_dict = {
        'state_dict': model.state_dict(),
        'state_mean': state_mean,
        'state_std': state_std,
    }
    torch.save(save_dict, output_path)
    print(f"Model saved to {output_path}")
    
    return model, final_dyn_loss, final_cbf_loss
