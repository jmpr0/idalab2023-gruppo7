import numpy as np
from torch import nn
import torch.nn.functional as F


class ShallowAutoencoder(nn.Module):
    """AutoEncoder with one dense layer for encoding and deconding"""
    
    def __init__(self, num_classes=10, **kwargs):
        super().__init__()

        ext_size = kwargs['num_pkts']*kwargs['num_fields']
        inner_size = int(np.ceil(ext_size*0.75))
        
        self.encoder = nn.Sequential(
            nn.Flatten(), 
            nn.Sigmoid(),
            nn.Linear(ext_size, inner_size),
        )
        self.decoder = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(inner_size, ext_size), 
            nn.Sigmoid()
        )
        # last classifier layer (head) with as many outputs as classes
        self.fc = nn.Linear(inner_size, num_classes)
        # and `head_var` with the name of the head, so it can be removed when doing incremental learning experiments
        self.head_var = 'fc'
        
    def forward(self, x):
        out = F.relu(self.extract_features(x))
        out = self.fc(out)
        return out
    
    def extract_features(self, x):
        out = self.encoder(x) 
        return out
    
    def extract_reconstruction(self, x):
        out = self.decoder(x)
        return out