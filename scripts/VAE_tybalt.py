import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision.transforms import ToTensor

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence
import torchvision
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import random
from random import sample
import seaborn as sns

class VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: list, z_dim):
        super(VAE, self).__init__()
        
        self.z_dim = z_dim
        self.input_dim = input_dim
        self.encoder_layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim[0])])
        self.decoder_layers = nn.ModuleList([nn.Linear(hidden_dim[0], input_dim)])
                
        if len(hidden_dim)>1:
            for i in range(len(hidden_dim)-1):
                self.encoder_layers.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
                self.decoder_layers.insert(0, nn.Linear(hidden_dim[i+1], hidden_dim[i]))
                
        self.encoder_layers.append(nn.Linear(hidden_dim[-1], 2 * z_dim))
        self.batchnorm = nn.BatchNorm1d(z_dim)
        self.decoder_layers.insert(0, nn.Linear(z_dim, hidden_dim[-1]))

        
    def encoder(self, x):
        for idx, layer in enumerate(self.encoder_layers):
            x = layer(x)
            if idx < len(self.encoder_layers) - 1:
                # x = F.dropout(x, 0.01)
                x = F.relu(x)
                #x = nn.BatchNorm1d(x)
        return x[...,:self.z_dim], x[...,self.z_dim:] # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        # std = torch.abs(log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        for idx, layer in enumerate(self.decoder_layers):
            z = layer(z)
            if idx < len(self.decoder_layers) - 1:
                # x = F.dropout(x, 0.01)
                z = F.relu(z)
        return torch.sigmoid(z) 
    
    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, self.input_dim))
        mu = self.batchnorm(mu)
        log_var = self.batchnorm(log_var)
    #    z = self.sampling(mu, log_var)
        latent = MultivariateNormal(loc = mu, 
                                    scale_tril=torch.diag_embed(torch.exp(0.5*log_var)))
        z = latent.rsample()
           
    #    return self.decoder(z), mu, log_var
        return self.decoder(z), latent

    @staticmethod
    def loss_function(recon_x, x, mu, log_var):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, input_dim), reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + KLD
    
    @staticmethod
    def loss_function_dist(recon_x, x, latent, input_dim):
        prior = MultivariateNormal(loc = torch.zeros(latent.mean.shape[1]),
                                   scale_tril=torch.eye(latent.mean.shape[1]))
        
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, input_dim), reduction='sum')
        KLD = torch.sum(kl_divergence(latent, prior))
        return BCE + KLD