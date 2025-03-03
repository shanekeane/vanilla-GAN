import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    Generator for GAN
    
    Input: noise_dim - length of input noise (int)
   
    """   

    def __init__(self, noise_dim):
        super(Generator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(noise_dim, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Linear(50, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid())
    
    def forward(self, x):
        return self.discriminator(x)

class Discriminator(nn.Module):
    """
    Discriminator for GAN.   
    """   
    def __init__(self):
        super(Discriminator, self).__init__()
        self.generator = nn.Sequential(
            nn.Linear(784, 200),
            nn.Dropout(0.3),
            nn.LeakyReLU(0.3),
            nn.Linear(200, 100),
            nn.Dropout(0.3),
            nn.LeakyReLU(0.3),
            nn.Linear(100, 1),
            nn.Sigmoid()) 

    def forward(self, x):
        return self.generator(x)
