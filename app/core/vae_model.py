# app/core/vae_model.py
# Description: This file contains the VAE model architecture definition.
# It's copied directly from your training script to ensure consistency.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class GELU(torch.nn.Module):
    """A wrapper for the GELU activation function."""
    def forward(self, input_tensor):
        return F.gelu(input_tensor)

class Encoder(nn.Module):
    """The VAE Encoder class."""
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super(Encoder, self).__init__()
        
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                GELU(),
                nn.Dropout(0.30)
            ])
            current_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        feature_dim = hidden_dims[-1] if hidden_dims else input_dim
        
        self.mu_net = nn.Linear(feature_dim, latent_dim)
        self.logvar_net = nn.Linear(feature_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(x)
        mu = self.mu_net(features)
        log_var = self.logvar_net(features)
        return mu, log_var

class Decoder(nn.Module):
    """The VAE Decoder class."""
    def __init__(self, latent_dim: int, hidden_dims: List[int], output_dim: int, output_activation: str = 'none'):
        super(Decoder, self).__init__()
        
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dims[0])
        
        layers = []
        # Note: The original training script had a potential bug here if len(hidden_dims) <= 1.
        # This implementation assumes len(hidden_dims) > 1 as in your training setup.
        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.BatchNorm1d(hidden_dims[i + 1]),
                GELU(),
                nn.Dropout(0.30)
            ])
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.reconstruction_net = nn.Sequential(*layers)
        self.output_activation = output_activation

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.latent_to_hidden(z))
        x_recon = self.reconstruction_net(hidden)
        
        if self.output_activation == 'sigmoid':
            x_recon = torch.sigmoid(x_recon)
        elif self.output_activation == 'tanh':
            x_recon = torch.tanh(x_recon)
            
        return x_recon

class VAE(nn.Module):
    """The complete VAE model class."""
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int, output_activation: str = 'none'):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        # The decoder's hidden dimensions are the reverse of the encoder's
        decoder_hidden_dims = hidden_dims[::-1]
        self.decoder = Decoder(latent_dim, decoder_hidden_dims, input_dim, output_activation)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """The reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(mu)
        return mu + std * eps

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The forward pass for the VAE."""
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var
