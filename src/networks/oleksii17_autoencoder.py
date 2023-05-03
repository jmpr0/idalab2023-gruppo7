import torch.nn as nn
import torch.nn.functional as F


class Oleksii17Autoencoder(nn.Module):
    def __init__(self, num_classes=10, **kwargs):
        super().__init__()

        ext_size = kwargs['num_pkts']*kwargs['num_fields']
        inner_size = 20

        self.encoder = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(ext_size, 10),
            nn.SELU(),
            nn.Linear(10, 10),
            nn.SELU(),
            nn.Linear(10, inner_size),
            nn.SELU(),
            nn.Dropout(0.8, inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(inner_size, 10),
            nn.SELU(),
            nn.Linear(10, 10),
            nn.SELU(),
            nn.Linear(10, ext_size),
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