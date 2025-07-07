import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from torchmetrics.functional import explained_variance
import numpy as np
from torch.nn.utils import spectral_norm

# If you are using the abstract_interpretation package:
from abstract_interpretation.neural_network import LinearLayer, ReLULayer, TanhLayer, NeuralNetwork

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
    For an input state s, it computes:
         z = [ s ; g_theta(s) ]
    where g_theta is a nonlinear feature extractor.
    """
    def __init__(self, state_dim, embed_dim):
        super(StateEmbedding, self).__init__()
        # Using the abstract_interpretation library's NeuralNetwork module.
        self.embed_net = nn.Sequential(
            nn.Linear(state_dim, 512), 
            # nn.BatchNorm1d(512),
            nn.SiLU(), 
            nn.Linear(512, 512),
            # nn.BatchNorm1d(512),
            nn.SiLU(), 
            nn.Linear(512, 512),
            # nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Linear(512, embed_dim),
            nn.Tanh()
            # TanhLayer()
        )
        self.state_dim = state_dim

    def forward(self, s):
        # s shape: (batch, state_dim)
        features = self.embed_net(s)  # shape: (batch, embed_dim)
        z = torch.cat([s, features], dim=-1)  # shape: (batch, state_dim + embed_dim)
        return z
    
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
        # self.c.weight.data.fill_(0.0)  # Initialize c to zero
        # self.c.bias.data.fill_(0.0)    # Initialize c to zero
        # self.c.require_grad = False  # Disable gradient updates for c

    def forward(self, z, u):
        # z: (batch_size, embed_total_dim)
        # u: (batch_size, control_dim)
        # return self.A(z) + self.B(u) + self.c(torch.cat([z, u], dim=-1))
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


##########################################################
# 3. PyTorch Lightning Module for Multi-Step Koopman Learning #
##########################################################

class KoopmanLightning(pl.LightningModule):
    """
    A Lightning Module that implements multi-step training of a Koopman model.
    It takes a sequence of states and a sequence of actions, then predicts the future latent states.
    The training objective is to match the multi-step latent trajectory with the encoder outputs
    from the corresponding ground truth future states.
    """
    def __init__(self, state_dim, embed_dim, control_dim, horizon, lr=0.001):
        """
        Args:
            state_dim (int): Dimensionality of the original state.
            embed_dim (int): Dimensionality of the nonlinear features.
            control_dim (int): Dimensionality of the control input.
            horizon (int): Prediction horizon (number of steps).
            lr (float): Learning rate.
        """
        super(KoopmanLightning, self).__init__()
        self.save_hyperparameters()
        self.state_dim = state_dim
        self.embed_dim = embed_dim
        self.control_dim = control_dim
        self.horizon = horizon
        self.lr = lr
        
        # Total embedding dimension, where z = [x; g_theta(x)]
        embed_total_dim = state_dim + embed_dim
        
        self.embedding_net = StateEmbedding(state_dim, embed_dim)
        self.koopman_operator = KoopmanOperator(embed_total_dim, control_dim)
        self.criterion = nn.MSELoss()
        self.mean = None
        self.std = None

    def forward(self, states, actions):
        """
        Forward pass for multi-step prediction.
        
        Args:
            states: Tensor of shape (B, horizon+1, state_dim), where the first state is s_0.
            actions: Tensor of shape (B, horizon, control_dim), corresponding to actions [u_0, ..., u_{horizon-1}].
        
        Returns:
            pred_latents: Predicted latent states for the next horizon steps,
                          a tensor of shape (B, horizon, embed_total_dim).
        """
        
        # Use the first state of each trajectory to initialize.
        s0 = states[:, 0, :]         # shape: (B, state_dim)
        z = self.embedding_net(s0)   # shape: (B, embed_total_dim)
        pred_latents = []
        for t in range(self.horizon):
            a_t = actions[:, t, :]   # shape: (B, control_dim)
            z = self.koopman_operator(z, a_t)  # shape: (B, embed_total_dim)
            pred_latents.append(z.unsqueeze(1))
        pred_latents = torch.cat(pred_latents, dim=1)  # shape: (B, horizon, embed_total_dim)
        return pred_latents

    def training_step(self, batch, batch_idx):
        states = batch['states']
        actions = batch['actions']
        next_states = batch['next_states']

        # 1. Multi-step prediction: pred_latents, shape (B, horizon, embed_total_dim)
        pred_latents = self.forward(states, actions)

        # 2. Ground truth latents: embedding of next_states (B, horizon, state_dim)
        B, H, S_dim = next_states.shape
        target_s_flat = next_states.reshape(B * H, -1)
        with torch.no_grad():
            target_latents_flat = self.embedding_net(target_s_flat)
            target_latents = target_latents_flat.reshape(B, H, -1)

        # 3. Main prediction loss (MSE in latent space)
        pred_loss = (pred_latents - target_latents).pow(2).mean()

        # 4. Consistency loss (autoencoding regularizer):
        #    For each pred_latent, decode and re-encode. Decoding: just [:, :, :state_dim]
        decoded = pred_latents[:, :, :S_dim]  # (B, horizon, state_dim)
        decoded_flat = decoded.reshape(B * H, -1)
        with torch.no_grad():
            reencoded_flat = self.embedding_net(decoded_flat)
            reencoded = reencoded_flat.reshape(B, H, -1)
        consistency_loss = (reencoded - pred_latents).pow(2).mean()

        # 5. Eigenvalue penalty (already in your code)
        A_w = self.koopman_operator.A.weight
        eigs = torch.linalg.eigvals(A_w)
        mags = torch.abs(eigs)
        excess = torch.clamp(mags - 1.0, min=0.0)
        eig_penalty = torch.sum(excess ** 2)

        # 6. Total loss (tune weights as needed)
        consistency_weight = 0.5
        eig_penalty_weight = 0.1
        # zeros = self.embedding_net(torch.zeros(1, self.state_dim))
        # zero_loss = torch.sum(zeros ** 2)
        total_loss = pred_loss + consistency_weight * consistency_loss + eig_penalty_weight * eig_penalty

        self.log('pred', pred_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('con', consistency_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('eig', eig_penalty, prog_bar=True, on_step=False, on_epoch=True)
        self.log('loss', total_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('ev_score', explained_variance(
            pred_latents[:, :, :S_dim].flatten(), target_latents[:, :, :S_dim].flatten(), multioutput='uniform_average'),
            prog_bar=True, on_step=False, on_epoch=True)
        return total_loss


    def validation_step(self, batch, batch_idx):
        states = batch['states']
        actions = batch['actions']
        next_states = batch['next_states']

        pred_latents = self.forward(states, actions)
        B, H, S_dim = next_states.shape
        target_s_flat = next_states.reshape(B * H, -1)
        with torch.no_grad():
            target_latents_flat = self.embedding_net(target_s_flat)
            target_latents = target_latents_flat.reshape(B, H, -1)

        pred_loss = (pred_latents - target_latents).pow(2).mean()

        # Consistency (autoencoding) loss
        decoded = pred_latents[:, :, :S_dim]
        decoded_flat = decoded.reshape(B * H, -1)
        with torch.no_grad():
            reencoded_flat = self.embedding_net(decoded_flat)
            reencoded = reencoded_flat.reshape(B, H, -1)
        consistency_loss = (reencoded - pred_latents).pow(2).mean()

        # Eigenvalue penalty
        A_w = self.koopman_operator.A.weight
        eigs = torch.linalg.eigvals(A_w)
        mags = torch.abs(eigs)
        excess = torch.clamp(mags - 1.0, min=0.0)
        eig_penalty = torch.sum(excess ** 2)

        consistency_weight = 0.5
        eig_penalty_weight = 0.1
        total_loss = pred_loss + consistency_weight * consistency_loss + eig_penalty_weight * eig_penalty

        self.log('val_pred', pred_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_con', consistency_loss, prog_bar=True, on_step=False, on_epoch=True)
        # self.log('val_eig', eig_penalty, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_ev', explained_variance(
            pred_latents[:, :, :S_dim].flatten(), target_latents[:, :, :S_dim].flatten(), multioutput='uniform_average'),
            prog_bar=True, on_step=False, on_epoch=True)
        return total_loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
        }
    
    def transition(self, z=None, z_1=None, u=None):
        A, B, c = self.koopman_operator.get_koopman_operators(z, u)
        return None, None, A, B, c, None, None
        
    
    @torch.no_grad()
    def get_eps(self, z, u):
        
        return self.eps_net(z, u)
    
    def transform(self, x):
        """
        Transforms an input (or batch of inputs) using the embedding network.
        Args:
            x: Input array of shape (B, state_dim) or (state_dim,)
        Returns:
            A numpy array of the latent representation of shape (B, state_dim+embed_dim) or (state_dim+embed_dim,)
        """
        x = torch.Tensor(x)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.embedding_net(x).detach().cpu().numpy()
    
    def get_next_state(self, states, actions):
        """
        Given a sequence of states and actions, predict the future latent sequence.
        Args:
            states: Array of shape (B, state_dim)
            actions: Array of shape (B, action_dim)
        Returns:
            pred_latents: Predicted latent states of shape (B, horizon, state_dim+embed_dim)
        """
        states = torch.Tensor(states)
        actions = torch.Tensor(actions)

        z = self.embedding_net(states)   # shape: (B, embed_total_dim)
        z = self.koopman_operator(z, actions)   # shape: (B, embed_total_dim)
        
        return z.detach().cpu().numpy()

    def update_stats(self, mean, std):
        self.mean = mean
        self.std = std
def fit_koopman(states, actions, next_states, koopman_model, horizon, epochs=100, val_size=0.3):
    """
    Fit the Koopman model on multi-step trajectory data, splitting into train and validation sets.

    Args:
        states (np.ndarray): Array of shape (N, horizon+1, state_dim)
        actions (np.ndarray): Array of shape (N, horizon, action_dim)
        next_states (np.ndarray): Array of shape (N, horizon+1, state_dim)
        koopman_model (KoopmanLightning): The Lightning module for Koopman learning.
        horizon (int): The prediction horizon.
        epochs (int): Number of training epochs.
        val_size (float): Proportion of the dataset to use for validation (e.g., 0.3 for 30%).
    """
    dataset = KoopmanDataset(states, actions, next_states)
    total_size = len(dataset)
    val_len = int(total_size * val_size)
    train_len = total_size - val_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=13)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=13)

    trainer = pl.Trainer(max_epochs=epochs, accelerator="cpu", devices=1, check_val_every_n_epoch=5)
    trainer.fit(koopman_model, train_loader, val_loader)
    
    A_w = koopman_model.koopman_operator.A.weight
    eigs = torch.linalg.eigvals(A_w)
    mags = torch.abs(eigs)
    print(mags)
    

    torch.cuda.empty_cache()

    del trainer
    del train_loader
    del val_loader
    # del residuals_loader
