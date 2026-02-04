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
    # Detect if image based on state keys or shape
    # states shape is (N, *state_dim)
    obs_shape = states.shape[1:]
    is_image = len(obs_shape) == 3
    
    if is_image:
        print("Detected Image Data. Using VisualEncoder/Decoder and /255 Normalization.")
        # Normalize images to [0, 1]
        # We set mean=0, std=255 so that at runtime: (obs - 0) / 255 ensures [0, 1]
        state_mean = torch.zeros(obs_shape)
        state_std = torch.ones(obs_shape) * 255.0
        
        # If data was saved as float 0-255, this brings it to 0-1.
        # If data was already 0-1? We assume raw 0-255 from replay buffer.
        states = states / 255.0
        next_states = next_states / 255.0
        
        state_dim = obs_shape[0] # Channels? VisualEncoder takes in_channels
        # Note: VisualEncoder args are (latent_dim, in_channels). 
        # obs_shape is (C, H, W).
        
        encoder = VisualEncoder(latent_dim=latent_dim, in_channels=obs_shape[0])
        decoder = VisualDecoder(latent_dim=latent_dim, out_channels=obs_shape[0])
    
    else:
        print("Detected Tabular Data. Using TabularEncoder/Decoder and Standard Normalization.")
        # Normalize States
        state_mean = states.mean(dim=0)
        state_std = states.std(dim=0) + 1e-6
        
        states = (states - state_mean) / state_std
        next_states = (next_states - state_mean) / state_std
        
        state_dim = states.shape[1]
        encoder = TabularEncoder(state_dim=state_dim, latent_dim=latent_dim)
        decoder = TabularDecoder(latent_dim=latent_dim, state_dim=state_dim)

    # Re-create dataset with normalized data
    dataset = TensorDataset(states, actions, next_states, horizon_targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    control_dim = actions.shape[1]
    
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

    device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    
    # --- 3. Stage 1: Variational Koopman Learning (Dynamics) ---
    print("\n=== Stage 1: Training Dynamics ===")
    optimizer_dyn = optim.Adam(model.parameters(), lr=lr)
    
    # Enable gradients
    for param in model.parameters(): param.requires_grad = True

    final_dyn_loss = 0.0
    for epoch in range(epochs_dyn):
        total_loss = 0
        
        for s, u, s_next, _ in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs_dyn}", leave=False):
            s, u, s_next = s.to(device), u.to(device), s_next.to(device)
            
            # Use convenience forward method for Stage 1
            out = model(s, u, s_next)
            
            # Compute Loss
            loss, _ = stage1_loss(out, s, s_next, dynamics=model.dynamics)
            
            optimizer_dyn.zero_grad()
            loss.backward()
            optimizer_dyn.step()
            
            total_loss += loss.item()
            
        final_dyn_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}: Loss = {final_dyn_loss:.4f}")
        
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
            s, u, y_h = s.to(device), u.to(device), y_h.to(device)
            y_h = y_h.unsqueeze(1) # (Batch, 1)
            
            # Encode & Predict (No Grads for dynamics)
            with torch.no_grad():
                mu_t, _ = model.encode(s)
                # Assume Im(z)=0 at start
                # Using mean for calculation
                z_re = mu_t
                z_im = torch.zeros_like(mu_t)
                z_next_re, z_next_im = model.predict_next(z_re, z_im, u)
                
            # CBF Score (Grads flow through w, beta)
            V = model.cbf_score(z_next_re, z_next_im)
            
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
