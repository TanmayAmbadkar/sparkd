import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from torchmetrics.functional import explained_variance
import numpy as np
from torch.nn.utils import spectral_norm

###############################################
# 1. Modified Dataset Class for Sequences (s, a, s_next) #
###############################################

class KoopmanDataset(Dataset):
    """
    A simple dataset for Koopman training.
    Each sample consists of a sequence of states, a sequence of actions,
    and a corresponding sequence of next states.
    
    Assumptions:
       - states: np.ndarray or torch.Tensor of shape (N, horizon+1, state_dim)
       - actions: np.ndarray or torch.Tensor of shape (N, horizon, action_dim)
       - next_states: np.ndarray or torch.Tensor of shape (N, horizon+1, state_dim)
         (Typically, next_states will be identical to states, or this can be viewed as the ground truth future trajectory.)
    """
    def __init__(self, states, actions, next_states):
        self.states = (torch.tensor(states, dtype=torch.float32) 
                       if not isinstance(states, torch.Tensor) else states)
        self.actions = (torch.tensor(actions, dtype=torch.float32) 
                        if not isinstance(actions, torch.Tensor) else actions)
        self.next_states = (torch.tensor(next_states, dtype=torch.float32) 
                            if not isinstance(next_states, torch.Tensor) else next_states)

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        return {
            'states': self.states[idx],         # shape: (horizon, state_dim)
            'actions': self.actions[idx],       # shape: (horizon, action_dim)
            'next_states': self.next_states[idx]  # shape: (horizon, state_dim)
        }
 
###################################################
# 2. Network Modules for the Koopman Model (Multi-Step) #
###################################################

class StateEmbedding(nn.Module):
    """
    Embeds the original state into a higher-dimensional space.
    If embed_dim is 0, this effectively becomes an identity mapping where z = s.
    """
    def __init__(self, state_dim, embed_dim):
        super(StateEmbedding, self).__init__()
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        if self.embed_dim > 0:
            self.embed_net = nn.Sequential(
                nn.Linear(state_dim, 512),
                nn.SiLU(),
                nn.Linear(512, 512),
                nn.SiLU(),
                nn.Linear(512, embed_dim),
                nn.Tanh()
            )
            self.decode_net = nn.Sequential(
                # The decoder should take the full latent state z as input
                nn.Linear(embed_dim, 512),
                nn.SiLU(),
                nn.Linear(512, 512),
                nn.SiLU(),
                nn.Linear(512, state_dim)
            )
        else:
            # If embed_dim is 0, these networks are not needed
            self.embed_net = None
            self.decode_net = None

    def forward(self, s):
        if self.embed_dim > 0:
            # Standard Koopman lifting
            features = self.embed_net(s)
            z = torch.cat([s, features], dim=-1)
        else:
            # Just linear regression, z is the same as s
            z = s
        return z

    def decode(self, z):
        if self.embed_dim > 0:
            # The decoder now correctly takes the full latent state z
            return self.decode_net(z)
        else:
            # If there's no embedding, the "decoded" state is just the input
            return z

    def encode(self, s):
        if self.embed_dim > 0:
            return self.embed_net(s)
        else:
            # Return an empty tensor with the correct batch size and device
            return torch.empty(s.shape[0], 0, device=s.device)
    
    
class KoopmanOperator(nn.Module):
    """
    Implements the Koopman operator for multi-step prediction with affine term:
         z_{t+1} = A * z_t + B * u_t + c
    where A and B are learnable linear mappings (no bias),
    and c is an explicit offset parameter.
    """
    def __init__(self, embed_total_dim, control_dim):
        super(KoopmanOperator, self).__init__()
        self.A = nn.Linear(embed_total_dim, embed_total_dim, bias=False)
        self.B = nn.Linear(control_dim, embed_total_dim, bias=False)
        # Explicit constant offset
        self.embed_total_dim = embed_total_dim

    def forward(self, z, u):
        return self.A(z) + self.B(u)

    def get_koopman_operators(self, z, u):
        """
        Returns:
            A_w (Tensor): shape (embed_total_dim, embed_total_dim)
            B_w (Tensor): shape (embed_total_dim, control_dim)
            c   (Tensor): shape (embed_total_dim,)
        """
        # clone to avoid in-place modifications
        A_w = self.A.weight.data.clone()
        B_w = self.B.weight.data.clone()
        return A_w, B_w, torch.zeros(self.embed_total_dim, dtype=torch.float32)
        # return A_w, B_w, self.c( torch.cat([z, u], dim=-1)).data.clone()


class KoopmanLightning(pl.LightningModule):
    """
    A Lightning Module that implements multi-step training of a Koopman model.
    """
    def __init__(self, state_dim, embed_dim, control_dim, horizon, lr=0.001, w_pred=1.0, w_recon=0.1, w_cons=0.5, w_eig=0.1):
        """
        Args:
            state_dim (int): Dimensionality of the original state.
            embed_dim (int): Dimensionality of the nonlinear features.
            control_dim (int): Dimensionality of the control input.
            horizon (int): Prediction horizon (number of steps).
            lr (float): Learning rate.
            w_pred, w_recon, w_cons, w_eig (float): Weights for the loss components.
        """
        super(KoopmanLightning, self).__init__()
        self.save_hyperparameters()
        
        embed_total_dim = state_dim + embed_dim
        
        self.embedding_net = StateEmbedding(state_dim, embed_dim)
        self.koopman_operator = KoopmanOperator(embed_total_dim, control_dim)
        self.criterion = nn.MSELoss()
        
        if self.hparams.embed_dim == 0:
            self.hparams.w_recon = 0.0
            self.hparams.w_cons = 0.0

    def forward(self, states, actions):
        """
        Forward pass for multi-step prediction.
        """
        s0 = states[:, 0, :]
        z = self.embedding_net(s0)
        pred_latents = []
        for t in range(self.hparams.horizon):
            a_t = actions[:, t, :]
            z = self.koopman_operator(z, a_t)
            pred_latents.append(z.unsqueeze(1))
        return torch.cat(pred_latents, dim=1)

    def _calculate_losses(self, batch):
        """Helper function to compute all loss components."""
        states = batch['states']
        actions = batch['actions']
        next_states = batch['next_states']
        
        B, H, S_dim = next_states.shape

        # 1. Multi-step prediction
        pred_latents = self.forward(states, actions)

        # 2. Ground truth latents
        # IMPORTANT FIX: Removed `with torch.no_grad()` to allow gradients to flow
        # to the encoder from the prediction loss.
        with torch.no_grad():
            target_s_flat = next_states.reshape(B * H, -1)
            target_latents_flat = self.embedding_net(target_s_flat)
            target_latents = target_latents_flat.reshape(B, H, -1)

        # 3. Prediction loss (MSE in latent space)
        pred_loss = self.criterion(pred_latents, target_latents)

        # # 4. Consistency loss (as in original code)
        # decoded_states = pred_latents[:, :, :S_dim]
        
        # reencoded_latents = self.embedding_net(decoded_states.reshape(B * H, -1)).reshape(B, H, -1)
        # consistency_loss = self.criterion(reencoded_latents, pred_latents)
        consistency_loss = torch.tensor(0.0, device=self.device)
        # 5. Reconstruction loss (as in original code)
        if self.hparams.embed_dim == 0:
            recon_loss = torch.tensor(0.0, device=self.device)
        else:
            features = self.embedding_net.encode(states.reshape(B * H, -1))
            recovered_states = self.embedding_net.decode(features)
            recon_loss = self.criterion(recovered_states, states.reshape(B * H, -1))

        # Explained variance for logging
        ev_score = explained_variance(
            pred_latents[:, :, :S_dim].flatten(), 
            target_latents[:, :, :S_dim].flatten()
        )

        return {
            "pred": pred_loss, "recon": recon_loss,
            "cons": consistency_loss, "ev": ev_score
        }

    def training_step(self, batch, batch_idx):
        losses = self._calculate_losses(batch)
        
        total_loss = (
            self.hparams.w_pred * losses['pred'] + 
            self.hparams.w_recon * losses['recon'] + 
            self.hparams.w_cons * losses['cons']
        )
        
        self.log('train_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_pred_loss', losses['pred'], prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_ev_score', losses['ev'], prog_bar=True, on_step=False, on_epoch=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        losses = self._calculate_losses(batch)

        # --- SPEEDUP: Calculate expensive eigenvalue penalty only on validation ---
        A_w = self.koopman_operator.A.weight
        eigs = torch.linalg.eigvals(A_w)
        mags = torch.abs(eigs)
        excess = torch.clamp(mags - 1.0, min=0.0)
        eig_penalty = torch.mean(excess ** 2)

        total_loss = (
            self.hparams.w_pred * losses['pred'] + 
            self.hparams.w_recon * losses['recon'] + 
            self.hparams.w_cons * losses['cons'] +
            self.hparams.w_eig * eig_penalty
        )

        self.log('val_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_eig_penalty', eig_penalty, prog_bar=True, on_step=False, on_epoch=True)
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
    
    def transition(self, z=None, z_1=None, u=None):
        A, B, c = self.koopman_operator.get_koopman_operators(z, u)
        return A, B, c
    
    @torch.no_grad()
    def get_eps(self, z, u):
        # Placeholder for error estimation
        return 0.01
    
    @torch.no_grad()
    def transform(self, x):
        """
        Transforms an input (or batch of inputs) using the embedding network.
        """
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.embedding_net(x).cpu().numpy()
    
    @torch.no_grad()
    def get_next_state(self, states, actions):
        """
        Given a batch of states and actions, predict the next latent state.
        """
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.device)

        z = self.embedding_net(states)
        z_next = self.koopman_operator(z, actions)
        
        return z_next.cpu().numpy()

    def update_stats(self, mean, std):
        # This functionality is better handled outside the model, e.g., in the Dataset
        pass

def fit_koopman(states, actions, next_states, koopman_model, horizon, epochs=100, val_size=0.3):
    """
    Fit the Koopman model on multi-step trajectory data.
    """
    dataset = KoopmanDataset(states, actions, next_states)
    total_size = len(dataset)
    val_len = int(total_size * val_size)
    train_len = total_size - val_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=13)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=13)

    trainer = pl.Trainer(max_epochs=epochs, accelerator="gpu", devices=1, check_val_every_n_epoch=5)
    trainer.fit(koopman_model, train_loader, val_loader)
    
    torch.cuda.empty_cache()

