import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from abstract_interpretation.neural_network import LinearLayer, ReLULayer, TanhLayer,SigmoidLayer, NeuralNetwork

from torch.nn import TripletMarginLoss

class TripletDataset(Dataset):
    """
    Custom dataset for loading observations and associated costs, returning triplets for triplet loss.
    
    Parameters:
    data (numpy.ndarray): The observations data.
    costs (list): The binary costs associated with the observations.
    """
    def __init__(self, data, costs):
        self.data = data
        self.costs = np.array(costs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a triplet (anchor, positive, negative).
        """
        anchor = torch.tensor(self.data[idx], dtype=torch.float32)
        anchor_cost = self.costs[idx]
        
        # Find a positive example (same cost as anchor)
        positive_idx = np.random.choice(np.where(self.costs == anchor_cost)[0])
        positive = torch.tensor(self.data[positive_idx], dtype=torch.float32)

        # Find a negative example (different cost than anchor)
        negative_idx = np.random.choice(np.where(self.costs != anchor_cost)[0])
        negative = torch.tensor(self.data[negative_idx], dtype=torch.float32)
        
        return (anchor, positive, negative)

class Autoencoder(pl.LightningModule):
    """
    Autoencoder neural network for dimensionality reduction with triplet loss.
    
    Parameters:
    n_features (int): The number of features in the input data.
    reduced_dim (int): The number of dimensions for the reduced representation.
    """
    def __init__(self, n_features, reduced_dim, margin=1.0):
        super().__init__()
        self.n_features = n_features
        self.reduced_dim = reduced_dim
        self.encoder = NeuralNetwork([LinearLayer(n_features, 12), TanhLayer(), LinearLayer(12, reduced_dim), TanhLayer()])
        self.decoder = NeuralNetwork([LinearLayer(reduced_dim, 12), TanhLayer(), LinearLayer(12, n_features)])
        self.loss_fn = TripletMarginLoss(margin=margin)

    def forward(self, x):
        """
        Forward pass through the autoencoder.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        """
        Encode the input data to its latent representation.
        """
        return self.encoder(x)
    
    def transform(self, x):
        """
        Transform the input data to its encoded representation.
        
        Parameters:
        x (numpy.ndarray): The input data.
        
        Returns:
        numpy.ndarray: The encoded data.
        """
        x = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            x_enc = self.encoder(x)
        return x_enc.numpy()

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step with triplet loss.
        """
        anchor, positive, negative = batch
        anchor_encoded = self.encode(anchor)
        positive_encoded = self.encode(positive)
        negative_encoded = self.encode(negative)
        
        # Calculate triplet margin loss
        triplet_loss = self.loss_fn(anchor_encoded, positive_encoded, negative_encoded)
        
        # Regular reconstruction loss for autoencoder
        # reconstructed_anchor = self(anchor)
        # reconstruction_loss = F.mse_loss(reconstructed_anchor, anchor)
        
        # Combine losses
        loss = triplet_loss
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer.
        """
        return torch.optim.Adam(self.parameters(), lr=0.001)

# Fit the Autoencoder with Triplet Loss
def fit_encoder(observations, costs, autoencoder):
    """
    Fit the autoencoder with triplet margin loss to the observations and costs.
    
    Parameters:
    observations (numpy.ndarray): The observations data.
    costs (list): The binary costs associated with the observations.
    autoencoder (TripletAutoencoder): The autoencoder model to be trained.
    """
    dataset = TripletDataset(observations, costs)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Initialize the trainer
    trainer = pl.Trainer(max_epochs=10, accelerator="cpu")

    # Train the autoencoder
    trainer.fit(autoencoder, train_loader)
