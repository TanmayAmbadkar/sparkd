import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from e2c.networks import Encoder, Transition, Decoder

class E2CPredictor(pl.LightningModule):
    def __init__(self, n_features, z_dim, u_dim, lr=1e-3, weight_decay=1e-4):
        super(E2CPredictor, self).__init__()
        self.encoder = Encoder(n_features, z_dim)
        transition_net = nn.Sequential(
            nn.Linear(z_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
        )
        self.transition = Transition(transition_net, z_dim, u_dim)
        self.decoder = Decoder(z_dim, n_features)
        self.z_dim = z_dim
        self.u_dim = u_dim
        self.lr = lr
        self.weight_decay = weight_decay

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
        x = x.view(-1, self.encoder.obs_dim).double().to(self.device)
        u = u.double().to(self.device)
        
        x_next = x_next.view(-1, self.encoder.obs_dim).double().to(self.device)

        # Forward pass
        x_recon, x_next_pred, z_t_next, z_t_pred = self._forward_step(x, u, x_next)


        # Compute loss
        lamda = 0.25  # You can parameterize this value
        loss = self.compute_loss(x, x_next, x_recon, x_next_pred, z_t_next, z_t_pred, lamda)

        # Log training loss
        self.log('train_loss', loss)
        return loss

    def _forward_step(self, x, u, x_next):
        # Step 1: Encode the current state x to get latent z_t
        z_t = self.encoder(x)
        
        # Step 2: Get the latent state z_t and predict A_t, B_t, o_t
        z_t_next, A_t, B_t, o_t = self.forward(x, u)
        
        # Step 3: Reconstruct x and predict next x using the decoder and transition model
        x_recon = self.decoder(z_t)  # Reconstruct x from z_t
        x_next_pred = self.decoder(z_t_next)
        z_t_pred = self.encoder(x_next)
        
        # Return relevant outputs
        return x_recon, x_next_pred, z_t_next, z_t_pred

    def compute_loss(self, x, x_next, x_recon, x_next_pred,  z_t_next, z_t_pred, lamda):
        # Reconstruction loss for current state
        recon_term = torch.mean((x - x_recon) ** 2)
        
        # Prediction loss for next state
        pred_loss = torch.mean((x_next - x_next_pred) ** 2)

        # Consistency loss
        consistency_loss = torch.mean((z_t_next - z_t_pred) ** 2)

        return recon_term + pred_loss + lamda * consistency_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    
    def transform(self, x):
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
    def __init__(self, states, actions, next_states):
        self.states = states
        self.actions = actions
        self.next_states = next_states

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
def fit_e2c(states, actions, next_states, e2c_predictor):
    """
    Fit the autoencoder with triplet margin loss to the observations and costs.
    
    Parameters:
    observations (numpy.ndarray): The observations data.
    costs (list): The binary costs associated with the observations.
    autoencoder (TripletAutoencoder): The autoencoder model to be trained.
    """
    dataset = E2CDataset(states, actions, next_states)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Initialize the trainer
    trainer = pl.Trainer(max_epochs=10, accelerator="cpu")

    # Train the autoencoder
    trainer.fit(e2c_predictor, train_loader)
