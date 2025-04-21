import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
from pcc.pcc_model import PCC
from torch.utils.data import DataLoader
from pcc.losses import bernoulli, KL, entropy, gaussian, curvature, vae_bound, ae_loss
import numpy as np
import gymnasium

class PCCPredictor(pl.LightningModule):
    def __init__(self, x_dim: int, z_dim: int, u_dim:int, amortized:bool=False, lr:float=0.001, weight_decay:float=1e-4):
        """
        Lightning module wrapper for the PCC model.
        
        Args:
            x_dim (int): Dimensionality of the observation.
            z_dim (int): Dimensionality of the latent space.
            u_dim (int): Dimensionality of the action.
            amortized (bool): Whether to use amortized curvature.
            lr (float): Learning rate.
            weight_decay (float): Weight decay for the optimizer.
        """
        super(PCCPredictor, self).__init__()
        self.model = PCC(amortized=amortized, x_dim=x_dim, z_dim=z_dim, u_dim=u_dim)
        self.lr = lr
        self.weight_decay = weight_decay
        self.amortized = amortized
        
        # Loss hyper-parameters (you can tune these)
        self.lam_p = 1.0
        self.lam_c = 5.0
        self.lam_cur = 1.0
        self.vae_coeff = 0.01
        self.determ_coeff = 0.3
        self.delta = 0.1

    def forward(self, x, u, x_next):
        # Pass through the PCC model
        raise NotImplementedError
        

    def compute_loss(
        self,
        amortized,
        x,
        u,
        x_next,
        p_x_next,
        dist_p_x_next,
        q_z_backward,
        p_z,
        q_z_next,
        z_next,
        p_z_next,
        z,
        p_x,
        dist_p_x,
        p_x_next_determ,
        lam=(1.0, 8.0, 8.0),
        delta=0.1,
        vae_coeff=0.01,
        determ_coeff=0.3,
    ):
        # prediction and consistency loss
        pred_loss = -gaussian(x_next, dist_p_x_next) + KL(q_z_backward, p_z) - entropy(q_z_next) - gaussian(z_next, p_z_next)

        consis_loss = -entropy(q_z_next) - gaussian(z_next, p_z_next) + KL(q_z_backward, p_z)

        # curvature loss
        cur_loss = curvature(self.model, z, u, delta, amortized)

        # additional vae loss
        vae_loss = vae_bound(x, dist_p_x, p_z)

        # additional deterministic loss)
        determ_loss = F.mse_loss(x_next, p_x_next_determ)

        lam_p, lam_c, lam_cur = lam
        return (
            pred_loss,
            consis_loss,
            cur_loss,
            lam_p * pred_loss
            + lam_c * consis_loss
            + lam_cur * cur_loss
            + vae_coeff * vae_loss
            + determ_coeff * determ_loss,
        )
    

    def training_step(self, batch, batch_idx):
        x, u, x_next = batch
        # Flatten observations if needed9
        
        # Forward pass through PCC
        (p_x_next, dist_p_x_next, q_z_backward, p_z, q_z_next, z_next,
         p_z_next, z_sample, _, p_x, dist_p_x, p_x_next_determ) = self.model(x, u, x_next)
        
        # Use the loss computation from your train_pcc.py
        lam_tuple = (self.lam_p, self.lam_c, self.lam_cur)
        pred_loss, consis_loss, cur_loss, total_loss = self.compute_loss(
            amortized = self.amortized, 
            x = x, 
            u = u, 
            x_next = x_next,
            p_x_next = p_x_next, 
            dist_p_x_next=dist_p_x_next,
            q_z_backward = q_z_backward, 
            p_z = p_z, 
            q_z_next = q_z_next, 
            z_next = z_next, 
            p_z_next = p_z_next,
            z=z_sample, 
            p_x = p_x, 
            dist_p_x = dist_p_x,
            p_x_next_determ = p_x_next_determ,
            lam=lam_tuple, 
            delta=self.delta,
            vae_coeff=self.vae_coeff, 
            determ_coeff=self.determ_coeff
        )
        
        self.log("pred_loss", pred_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("consis_loss", consis_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("cur_loss", cur_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("total_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
        return total_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

class PCCDataset(Dataset):
    """
    Dataset for PCC training. Returns triplets: (state, action, next_state)
    """
    def __init__(self, states, actions, next_states):
        self.states = states
        self.actions = actions
        self.next_states = next_states
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        state = torch.tensor(self.states[idx], )
        action = torch.tensor(self.actions[idx], )
        next_state = torch.tensor(self.next_states[idx], )
        return state, action, next_state


def fit_pcc(states, actions, next_states, model,
            epochs=100, batch_size=512, lr=1e-3, weight_decay=1e-4, amortized=False):
    """
    Train a PCC model using PyTorch Lightning.
    
    Args:
        states (np.ndarray): Array of observations.
        actions (np.ndarray): Array of actions.
        next_states (np.ndarray): Array of next observations.
        x_dim (int): Observation dimension.
        z_dim (int): Latent dimension.
        u_dim (int): Action dimension.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        lr (float): Learning rate.
        weight_decay (float): Weight decay.
        amortized (bool): Whether to use amortized curvature.
        
    Returns:
        model (PCCPredictor): The trained PCC predictor.
    """
    dataset = PCCDataset(states, actions, next_states)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    trainer = pl.Trainer(max_epochs=epochs, accelerator="gpu", devices=1)
    trainer.fit(model, train_loader)
    return model
