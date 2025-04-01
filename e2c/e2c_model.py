import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from e2c.networks import Encoder, Decoder, Transition
from e2c.distribution import NormalDistribution
import copy


class E2CPredictor(pl.LightningModule):
    def __init__(self, n_features, z_dim, u_dim, lr=0.001, weight_decay=1e-4, horizon=1, train_ae=True, last_predictor = None):
        super(E2CPredictor, self).__init__()
        self.encoder = Encoder(n_features, z_dim)
        transition_net = nn.Sequential(
            nn.Linear(z_dim, 12),
            nn.ReLU(),
            nn.Linear(12, 12),
            nn.ReLU(),
        )
        self.transition = Transition(transition_net, z_dim, u_dim)
        self.decoder = Decoder(z_dim, n_features)
        self.z_dim = z_dim
        self.u_dim = u_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.horizon = horizon
        self.train_ae = train_ae
        self.mean = np.zeros(n_features)
        self.std = np.ones(n_features)
        self.last_predictor = last_predictor
        # Hyperparameters for curvature loss (PCC component)
        self.lambda_curvature = 1.0  # weight for curvature loss (tune as needed)
        self.delta = 0.1             # standard deviation for latent/state perturbations


    def forward(self, x_t, u_t=None):
        z_t, _, _ = self.encoder(x_t)
        if u_t is None:
            u_t = torch.zeros((x_t.size(0), self.u_dim)).to(x_t.device)
        z_t_next, A_t, B_t, o_t = self.transition(z_t, z_t, u_t)
        return z_t_next, A_t, B_t, o_t

    def training_step(self, batch, batch_idx):
        x, u, x_next = batch
        x = x.double()
        u = u.double()
        x_next = x_next.double()

        # --- Encoding ---
        z_t, mu, logsig = self.encoder(x)           # Current latent representation
        z_t_next_enc, mu_next, logsig_next = self.encoder(x_next)  # Next latent representation from encoder

        # --- Reconstruction of current observation ---
        x_recon = self.decoder(z_t)
        loss_recon = F.mse_loss(x_recon, x)

        # --- KL Divergence Loss (VAE loss) ---
        loss_kl = -0.5 * torch.mean(1 + 2 * logsig - mu.pow(2) - torch.exp(2 * logsig))

        # --- Transition / Prediction ---
        # Use transition network to predict next latent state and decode to next observation.
        z_t_next_pred, z_t_next_mean, A_t, B_t, _, v_t, r_t = self.transition(z_t, mu, u)
        x_next_pred = self.decoder(z_t_next_pred)
        loss_pred = F.mse_loss(x_next_pred, x_next)

        # --- Consistency Loss ---
        # Ensure that the latent dynamics prediction (transition distribution) matches the encoded next state.
        transition_distribution = NormalDistribution(z_t_next_mean, logsig, v_t.squeeze(), r_t.squeeze())
        next_distribution = NormalDistribution(mu_next, logsig_next)
        loss_consis = NormalDistribution.KL_divergence(transition_distribution, next_distribution)

        # # --- Curvature Loss ---
        # # Perturb the latent state and action to enforce local linearity.
        # eps_z = torch.randn_like(z_t) * self.delta
        # eps_u = torch.randn_like(u) * self.delta
        # z_t_perturbed = z_t + eps_z
        # u_perturbed = u + eps_u

        # # Compute the perturbed next latent state using the transition network.
        # # We use the same network structure; here we take the output mean.
        # with torch.no_grad():
        #     f_perturbed = self.transition(z_t_perturbed, z_t_perturbed, u_perturbed)[1]
        # # Linear approximation: f_nominal + A_t * eps_z + B_t * eps_u.
        # linear_approx = z_t_next_mean + (A_t.bmm(eps_z.unsqueeze(-1))).squeeze(-1) \
        #                               + (B_t.bmm(eps_u.unsqueeze(-1))).squeeze(-1)
        # loss_curvature = F.mse_loss(f_perturbed, linear_approx)

        # --- Combine Losses ---
        # Adjust the weights as needed; here consistency and curvature losses are weighted higher.
        total_loss = loss_recon + loss_kl + loss_pred + 5 * loss_consis

        # --- Logging ---
        self.log("loss_recon", loss_recon, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log("loss_kl", loss_kl, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log("loss_pred", loss_pred, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log("loss_consis", loss_consis, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        # self.log("loss_curvature", loss_curvature, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log("total_loss", total_loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)

        return total_loss



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    
    def transform(self, x):
        x = (x - self.mean) / self.std
        x = torch.tensor(x).double()
        # print(x)
        with torch.no_grad():
            z_t = self.encoder.encode(x)
        
        return z_t.numpy()
    
    def inverse_transform(self, x):
        x = torch.tensor(x).double()
        with torch.no_grad():
            z_t = self.decoder(x)
        
        z_t = z_t * self.std + self.mean
        
        return z_t.numpy()
    
    def get_next_state(self, x, u):
        # x = torch.tensor(x).double()
        # u = torch.tensor(u).double()
        with torch.no_grad():
            mu = self.encoder.encode(x)
            z_t_next, _, _, _, _, _, _ = self.transition(mu, mu, u)
            dec_state = self.decoder(z_t_next)

            # z_t_next = A_t.bmm(z_t.unsqueeze(-1)).squeeze(-1) + B_t.bmm(u.unsqueeze(-1)).squeeze(-1) + o_t
            
        return dec_state.numpy()

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

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        """
        Returns a triplet (anchor, positive, negative).
        """
        state = torch.tensor(self.states[idx], dtype=torch.float32)
        action = torch.tensor(self.actions[idx], dtype=torch.float32)
        next_state = torch.tensor(self.next_states[idx], dtype=torch.float32)
        
        return (state, action, next_state)
    
# Fit the Autoencoder with Triplet Loss
def fit_e2c(states, actions, next_states, e2c_predictor, horizon, epochs = 100):
    """
    Fit the autoencoder with triplet margin loss to the observations and costs.
    
    Parameters:
    observations (numpy.ndarray): The observations data.
    costs (list): The binary costs associated with the observations.
    autoencoder (TripletAutoencoder): The autoencoder model to be trained.
    """
    dataset = E2CDataset(states, actions, next_states, horizon)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=1)

    # Initialize the trainer
    e2c_predictor.train_ae = True
    trainer = pl.Trainer(max_epochs=epochs, accelerator="gpu", devices = 1)

    # Train the autoencoder
    trainer.fit(e2c_predictor, train_loader)
    # print(next(e2c_predictor.transition.parameters()))
    
    del trainer
    e2c_predictor.last_predictor = copy.deepcopy(e2c_predictor)

    print(np.round(e2c_predictor.get_next_state( torch.tensor(states[:10], dtype=torch.float64), torch.tensor(actions[:10], dtype=torch.float64)), 2))
    print(np.round(e2c_predictor.transform(next_states[:10]), 2))
    
    del train_loader
    
    