import torch
import torch.nn.functional as F
from torch import nn
import math

from .network import get_output_dim


class Jorgensen22MLP(nn.Module):
    """Wang17-like network used as a single modality by Aceto19 for tests with PAYLOAD input."""

    def __init__(self, in_channels=1, num_classes=10, **kwargs):
        super().__init__()

        num_pkts = kwargs['num_pkts']
        num_fields = kwargs['num_fields']
        out_features_size = kwargs['out_features_size']
        if out_features_size is None:
            out_features_size = 64
        
        scaling_factor = kwargs.get('scale', 1)
        n_neurons = math.ceil(64 * scaling_factor)

        # main part of the network
        self.fc4 = nn.Linear(num_pkts * num_fields, n_neurons)
        self.fc3 = nn.Linear(n_neurons, n_neurons)
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.fc1 = nn.Linear(n_neurons, out_features_size)

        # last classifier layer (head) with as many outputs as classes
        self.fc = nn.Linear(out_features_size, num_classes)
        # and `head_var` with the name of the head, so it can be removed when doing incremental learning experiments
        self.head_var = 'fc'

        self.activation = F.relu
        # self.activation = torch.tanh

    def forward(self, x):
        out = self.activation(self.extract_features(x))
        out = self.fc(out)
        return out

    def extract_features(self, x):
        out = torch.flatten(x, start_dim=1)
        out = self.activation(self.fc4(out))
        out = self.activation(self.fc3(out))
        out = F.dropout(out, 0.25)
        out = self.activation(self.fc2(out))
        out = F.dropout(out, 0.25)
        out = self.fc1(out)
        return out
