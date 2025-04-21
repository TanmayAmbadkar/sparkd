import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from torchmetrics.regression import ExplainedVariance

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
        self.embed_net = NeuralNetwork([
            LinearLayer(state_dim, embed_dim), 
            ReLULayer(), 
            LinearLayer(embed_dim, embed_dim),
            TanhLayer()
        ])
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
        self.c = nn.Parameter(torch.zeros(embed_total_dim))

    def forward(self, z, u):
        # z: (batch_size, embed_total_dim)
        # u: (batch_size, control_dim)
        return self.A(z) + self.B(u) + self.c

    def get_koopman_operators(self):
        """
        Returns:
            A_w (Tensor): shape (embed_total_dim, embed_total_dim)
            B_w (Tensor): shape (embed_total_dim, control_dim)
            c   (Tensor): shape (embed_total_dim,)
        """
        # clone to avoid in-place modifications
        A_w = self.A.weight.data.clone()
        B_w = self.B.weight.data.clone()
        c   = self.c.data.clone()
        return A_w, B_w, c


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
    def __init__(self, state_dim, embed_dim, control_dim, horizon, lr=1e-3):
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
        self.ev = ExplainedVariance()
        self.criterion = nn.MSELoss()

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
        B = states.shape[0]
        # Use the first state of each trajectory to initialize.
        s0 = states[:, 0, :]         # shape: (B, state_dim)
        z = self.embedding_net(s0)   # shape: (B, embed_total_dim)
        pred_latents = []
        for t in range(self.horizon):
            a_t = actions[:, t, :]   # shape: (B, control_dim
            z = self.koopman_operator(z, a_t)   # shape: (B, embed_total_dim)
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
            loss[:, i] = loss[:, i] * 0.95**i
        loss = loss.mean()
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('ev_score', self.ev(pred_latents, target_latents), prog_bar=True, on_step=False, on_epoch=True)
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
            loss[:, i] = loss[:, i] * 0.95**i
        loss = loss.mean()
        self.log('val_train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_ev_score', self.ev(pred_latents, target_latents), prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def transition(self, z=None, z_1=None, u=None):
        A, B, c = self.koopman_operator.get_koopman_operators()
        return None, None, A, B, c, None, None
    
    def transform(self, x):
        """
        Transforms an input (or batch of inputs) using the embedding network.
        Args:
            x: Input array of shape (B, state_dim) or (state_dim,)
        Returns:
            A numpy array of the latent representation of shape (B, state_dim+embed_dim) or (state_dim+embed_dim,)
        """
        x = torch.tensor(x)
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
        states = torch.tensor(states)
        actions = torch.tensor(actions)

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

    trainer = pl.Trainer(max_epochs=epochs, accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1, precision = 16)
    trainer.fit(koopman_model, train_loader, val_loader)
    torch.cuda.empty_cache()

    del trainer
    del train_loader
    del val_loader
