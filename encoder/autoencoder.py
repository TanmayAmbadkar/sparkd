import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from abstract_interpretation.neural_network import LinearLayer, ReLULayer, NeuralNetwork

# Define the Autoencoder Network
class Autoencoder(pl.LightningModule):
    """
    Autoencoder neural network for dimensionality reduction.
    
    Parameters:
    n_features (int): The number of features in the input data.
    reduced_dim (int): The number of dimensions for the reduced representation.
    """
    def __init__(self, n_features, reduced_dim):
        super().__init__()
        self.n_features = n_features
        self.reduced_dim = reduced_dim
        self.encoder = NeuralNetwork([LinearLayer(n_features, 12), ReLULayer(), LinearLayer(12, reduced_dim)])
        self.decoder = NeuralNetwork([LinearLayer(reduced_dim, 12), ReLULayer(), LinearLayer(12, n_features)])

    def forward(self, x):
        """
        Forward pass through the autoencoder.
        
        Parameters:
        x (torch.Tensor): The input data.
        
        Returns:
        torch.Tensor: The reconstructed data.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
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
        Perform a single training step.
        
        Parameters:
        batch (tuple): A batch of data.
        batch_idx (int): The index of the batch.
        
        Returns:
        torch.Tensor: The loss value for the batch.
        """
        x, _ = batch
        reconstructed = self(x)
        loss = nn.MSELoss()(reconstructed, x)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer for training.
        
        Returns:
        torch.optim.Optimizer: The optimizer.
        """
        return torch.optim.Adam(self.parameters(), lr=0.001)

# Create the Dataset
class CustomDataset(Dataset):
    """
    Custom dataset for loading observations.
    
    Parameters:
    data (numpy.ndarray): The observations data.
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        """
        Get the number of samples in the dataset.
        
        Returns:
        int: The number of samples.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Parameters:
        idx (int): The index of the sample.
        
        Returns:
        tuple: A tuple containing the sample and its corresponding label (the same sample).
        """
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.data[idx], dtype=torch.float32)

def fit_encoder(observations, autoencoder):
    """
    Fit the autoencoder to the observations data.
    
    Parameters:
    observations (numpy.ndarray): The observations data.
    autoencoder (Autoencoder): The autoencoder model to be trained.
    """
    dataset = CustomDataset(observations)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Initialize the trainer
    trainer = pl.Trainer(max_epochs=10, accelerator="cpu")

    # Train the autoencoder
    trainer.fit(autoencoder, train_loader)
