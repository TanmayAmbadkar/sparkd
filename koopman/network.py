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


class ResidualsDataset(Dataset):
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
    def __init__(self, states, actions, residuals):
        self.states = (torch.tensor(states, dtype=torch.float32) 
                       if not isinstance(states, torch.Tensor) else states)
        self.actions = (torch.tensor(actions, dtype=torch.float32) 
                        if not isinstance(actions, torch.Tensor) else actions)
        self.residuals = (torch.tensor(residuals, dtype=torch.float32) 
                       if not isinstance(residuals, torch.Tensor) else residuals)
    def __len__(self):
        return self.residuals.shape[0]

    def __getitem__(self, idx):
        return {
            'states': self.states[idx],         # shape: ( state_dim)
            'actions': self.actions[idx],   
            'residuals': self.residuals[idx],         # shape: (embed_total_dim)
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
            nn.Linear(state_dim, 256), 
            nn.ReLU(), 
            nn.Linear(256, 256),
            nn.ReLU(), 
            nn.Linear(256, embed_dim),
            # nn.Tanh()
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
        self.c = nn.Sequential(
            nn.Linear(embed_total_dim + control_dim, 256), 
            nn.ReLU(), 
            nn.Linear(256, 128),
            nn.ReLU(), 
            nn.Linear(128, embed_total_dim),
        )
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


class UncertaintyNet(pl.LightningModule):
    
    def __init__(self, state_dim, control_dim, quantile: float = 0.95, lr: float = 1e-3):
        super(UncertaintyNet, self).__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.quantile = quantile
        self.lr = lr
        self.net = nn.Sequential(
            nn.Linear(state_dim + control_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim),
            # nn.ReLU()
        )
        
    def forward(self, z, u):
        return self.net(torch.cat([z, u], dim=-1))
    
    def training_step(self, batch, batch_idx):
        z = batch['states']       # (B, state_dim)
        u = batch['actions']      # (B, control_dim)
        r_true = torch.abs(batch['residuals'])  # (B, state_dim+control_dim)
        
        r_pred = self(z, u)
        error = (r_true - r_pred)**2  # shape (B, state_dim+control_dim)
        
        # Pinball (quantile) loss
        # q = self.quantile
        # loss_matrix = torch.maximum(q * error, (q - 1) * error)**2
        loss = error.mean()
        
        self.log('loss', loss, prog_bar=True, on_epoch=True, on_step=True)
        self.log("ev", explained_variance(r_pred.flatten(), r_true.flatten() , multioutput='uniform_average'), prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


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
    def __init__(self, state_dim, embed_dim, control_dim, horizon, lr=0.0001):
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
        self.eps_net = None
        self.criterion = nn.MSELoss()
        self.z0 = None
        self.lip_const = None
        self.eps0 = None

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
            a_t = actions[:, t, :]   # shape: (B, control_dim
            z = self.koopman_operator(z, a_t)  # shape: (B, embed_total_dim)
            pred_latents.append(z.unsqueeze(1))
        pred_latents = torch.cat(pred_latents, dim=1)  # shape: (B, horizon, embed_total_dim)
        return pred_latents

    def training_step(self, batch, batch_idx):
        # Expected shapes:
        # batch['states']: (B, horizon+1, state_dim)
        # batch['actions']: (B, horizon, action_dim)
        # batch['next_states']: (B, horizon+1, state_dim)
        states = batch['states']
        actions = batch['actions']
        next_states = batch['next_states']
        
        # Corrected target calculation:
        # Target states are s_1, s_2, ..., s_horizon from the ground truth
        target_s = next_states # Shape: (B, horizon, state_dim)
        B, H, S_dim = target_s.shape

        # Reshape to (B*H, state_dim) for the embedding network
        target_s_flat = target_s.reshape(B * H, -1)
        

        with torch.no_grad():
            # Ensure embedding net is in eval mode if it has dropout/batchnorm, though yours doesn't seem to
            # self.embedding_net.eval() # Optional here as training_step implies train mode
            target_latents_flat = self.embedding_net(target_s_flat) # Use .float() assuming float32
            # self.embedding_net.train() # Optional: switch back if needed

            # Reshape back to (B, horizon, embed_total_dim)
            target_latents = target_latents_flat.reshape(B, H, -1)

        pred_latents = self.forward(states, actions)
        loss = (pred_latents - target_latents) ** 2
        for i in range(self.horizon):
            loss[:, i] = loss[:, i] * 0.99**i
        # loss = loss.mean() + pred_latents[:, :, self.state_dim:].pow(2).mean() * 1e-3
        loss = loss.mean()
         
        # # sparsity on Koopman weights
        # lam_A = 1e-4
        # loss += lam_A * (self.koopman_operator.A.weight.abs().sum() + self.koopman_operator.B.weight.abs().sum())

        # # sparsity on embedding net weights
        # lam_emb = 1e-5
        # for w in self.embedding_net.parameters():
        #     loss += lam_emb * w.abs().sum()
        
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('ev_score', explained_variance(pred_latents.flatten(), target_latents.flatten(), multioutput='uniform_average'), prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Expected shapes:
        # batch['states']: (B, horizon+1, state_dim)
        # batch['actions']: (B, horizon, action_dim)
        # batch['next_states']: (B, horizon+1, state_dim)
        states = batch['states']
        actions = batch['actions']
        next_states = batch['next_states']
        
        # Corrected target calculation:
        # Target states are s_1, s_2, ..., s_horizon from the ground truth
        target_s = next_states # Shape: (B, horizon, state_dim)
        B, H, S_dim = target_s.shape
        embed_total_dim = self.state_dim + self.embed_dim

        # Reshape to (B*H, state_dim) for the embedding network
        target_s_flat = target_s.reshape(B * H, -1)

        with torch.no_grad():
            # Ensure embedding net is in eval mode if it has dropout/batchnorm, though yours doesn't seem to
            # self.embedding_net.eval() # Optional here as training_step implies train mode
            target_latents_flat = self.embedding_net(target_s_flat) # Use .float() assuming float32
            # self.embedding_net.train() # Optional: switch back if needed

            # Reshape back to (B, horizon, embed_total_dim)
            target_latents = target_latents_flat.reshape(B, H, -1)

        pred_latents = self.forward(states, actions)
        loss = (pred_latents - target_latents) ** 2
        for i in range(self.horizon):
            loss[:, i] = loss[:, i] * 0.9**i
        loss = loss.mean()
        
               
        self.log('val_train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_ev_score', explained_variance(pred_latents.flatten(), target_latents.flatten() , multioutput='uniform_average'), prog_bar=True, on_step=False, on_epoch=True)
        return loss

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

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=1)

    trainer = pl.Trainer(max_epochs=epochs, accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1, check_val_every_n_epoch=5)
    trainer.fit(koopman_model, train_loader, val_loader)
    
    print(koopman_model.koopman_operator.A.weight)
    print(koopman_model.koopman_operator.B.weight)

    # with torch.no_grad():
        
    #     koopman_model.eps_net = UncertaintyNet(koopman_model.state_dim + koopman_model.embed_dim, koopman_model.control_dim) if koopman_model.eps_net is None else koopman_model.eps_net
    #     residuals = koopman_model.transform(dataset.next_states[:, 0]) - koopman_model.get_next_state(dataset.states[:, 0], dataset.actions[:, 0])
    #     residuals_dataset = ResidualsDataset(koopman_model.transform(dataset.states[:, 0]), dataset.actions[:, 0], residuals)
    #     residuals_loader = DataLoader(residuals_dataset, batch_size=128, shuffle=False, num_workers=1)

    #     trainer = pl.Trainer(max_epochs=epochs, accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1)
    #     trainer.fit(koopman_model.eps_net, residuals_loader)


    torch.cuda.empty_cache()

    del trainer
    del train_loader
    del val_loader
    # del residuals_loader
