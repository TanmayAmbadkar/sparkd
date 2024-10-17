import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader, Dataset
from e2c.networks import Encoder, Transition, Decoder

class E2CPredictor(pl.LightningModule):
    def __init__(self, n_features, z_dim, u_dim, lr=1e-3, weight_decay=1e-4, horizon = 1):
        super(E2CPredictor, self).__init__()
        self.encoder = Encoder(n_features, z_dim)
        transition_net = nn.Sequential(
            nn.Linear(z_dim, 12),
            nn.Tanh(),
            nn.Linear(12, 12),
            nn.Tanh(),
        )
        self.transition = Transition(transition_net, z_dim, u_dim)
        self.decoder = Decoder(z_dim, n_features)
        self.z_dim = z_dim
        self.u_dim = u_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.horizon = horizon
        self.states_mean = None
        self.states_std = None

    def forward(self, x_t, u_t=None):
        """
        :param x_t: the input observation at time t (e.g., image or state)
        :param u_t: the action taken at time t (can be set to zero if not used)
        :return: z_t, A_t, B_t, o_t (the latent state, state transition matrix, control matrix, and offset)
        """
        # Step 1: Encode x_t to get z_t
        z_t = self.encoder(x_t)

        # Step 2: Set u_t to a zero vector if not provided
        if u_t is None:
            u_t = torch.zeros((x_t.size(0), self.u_dim)).to(x_t.device)  # assuming batch size first

        # Step 3: Predict A_t, B_t, o_t using the transition model
        z_t_next, A_t, B_t, o_t = self.transition(z_t, u_t)

        # Return latent state z_t, and the matrices A_t, B_t, and offset o_t
        return z_t_next, A_t, B_t, o_t

    def training_step(self, batch, batch_idx):
        x, u, x_next = batch
        x = x.double()
        u = u.double()
        x_next = x_next.double()

        # Forward pass
        # x_recon, x_next_pred, z_t_next, z_t_pred = self._forward_step(x, u, x_next)

        z_t, z_t_next, ae_loss = self._forward_ae_step(x, x_next)
        transition_loss = self._forward_transition_step(z_t, u, z_t_next, x_next)
        # Compute loss
        lamda = 0.25  # You can parameterize this value
        # loss = self.compute_loss(x, x_next, x_recon, x_next_pred, z_t_next, z_t_pred, lamda)

        # Log training loss
        loss = ae_loss + transition_loss
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def _forward_ae_step(self, x, x_next):
        
        z_t = []
        z_t_next = []
        x_recon = []
        x_next_recon = []
        with torch.no_grad():
            for idx in range(x.shape[0]):
                z_t.append(self.encoder(x[idx]))
                z_t_next.append(self.encoder(x_next[idx]))
                x_recon.append(self.decoder(z_t[idx]))
                x_next_recon.append(self.decoder(z_t[idx]))
                
        z_t = torch.stack(z_t)
        z_t_next = torch.stack(z_t_next)
        x_recon = torch.stack(x_recon)
        x_next_recon = torch.stack(x_next_recon)
        
        return z_t, z_t_next, F.mse_loss(x[:,1], self.decoder(self.encoder(x[:,1]))) + F.mse_loss(x_next[:,1], self.decoder(self.encoder(x_next[:,1])))
        
    
    def _forward_transition_step(self, z_t, u, z_t_next, x_next):
        
        transition_loss = 0
        consistency_loss = 0
        for hor in range(self.horizon):
            z_t_curr = z_t[:, hor]
            _, A_t, B_t, o_t = self.transition(z_t_curr, u[:, hor])
            for idx in range(hor, self.horizon):
                
                z_t_next_pred = A_t.bmm(z_t_curr.unsqueeze(-1)).squeeze(-1) + B_t.bmm(u[:, idx].unsqueeze(-1)).squeeze(-1) + o_t
                
                transition_loss += F.mse_loss(z_t_next_pred, z_t_next[:, idx])
                consistency_loss += F.mse_loss(x_next[:, idx], self.decoder(z_t_next_pred))
                z_t_curr = z_t_next_pred
                
                
        return transition_loss + consistency_loss
                               

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    
    def transform(self, x):
        # x = (torch.tensor(x).double() - self.states_mean)/self.states_std
        x = torch.tensor(x).double()
        with torch.no_grad():
            z_t = self.encoder(x)
        
        return z_t.numpy()
    
    def get_next_state(self, x, u):
        x = torch.tensor(x).double()
        u = torch.tensor(u).double()
        with torch.no_grad():
            z_t_next, A_t, B_t, o_t = self.transition(x, u)

            # z_t_next = A_t.bmm(z_t.unsqueeze(-1)).squeeze(-1) + B_t.bmm(u.unsqueeze(-1)).squeeze(-1) + o_t
            
        return z_t_next.numpy()
    
    

class E2CDataset(Dataset):
    """
    Custom dataset for loading observations and associated costs, returning triplets for triplet loss.
    
    Parameters:
    data (numpy.ndarray): The observations data.
    costs (list): The binary costs associated with the observations.
    """
    def __init__(self, states, actions, next_states, horizon = 1):
        self.states = states
        self.actions = actions
        self.next_states = next_states
        self.horizon = horizon
        
        self.states_mean = np.concatenate((states, next_states),
                                 axis=0).mean(axis=0)
        self.states_std = np.maximum(np.concatenate((states, next_states),
                                           axis=0).std(axis=0), 1e-5)
        
        # self.states = (self.states - self.states_mean)/self.states_std
        # self.next_states = (self.next_states - self.states_mean)/self.states_std

    def __len__(self):
        return len(self.states) - self.horizon

    def __getitem__(self, idx):
        """
        Returns a triplet (anchor, positive, negative).
        """
        state = torch.tensor(self.states[idx:idx + self.horizon], dtype=torch.float32)
        action = torch.tensor(self.actions[idx:idx + self.horizon], dtype=torch.float32)
        next_state = torch.tensor(self.next_states[idx:idx + self.horizon], dtype=torch.float32)
        
        return (state, action, next_state)


# Fit the Autoencoder with Triplet Loss
def fit_e2c(states, actions, next_states, e2c_predictor, horizon):
    """
    Fit the autoencoder with triplet margin loss to the observations and costs.
    
    Parameters:
    observations (numpy.ndarray): The observations data.
    costs (list): The binary costs associated with the observations.
    autoencoder (TripletAutoencoder): The autoencoder model to be trained.
    """
    dataset = E2CDataset(states, actions, next_states, horizon)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    e2c_predictor.states_mean = dataset.states_mean
    e2c_predictor.states_std = dataset.states_std

    # Initialize the trainer
    trainer = pl.Trainer(max_epochs=20, accelerator="cpu")

    # Train the autoencoder
    trainer.fit(e2c_predictor, train_loader)
