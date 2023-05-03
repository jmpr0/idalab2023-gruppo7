import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math


class Lopez17RNN(nn.Module):
    """Lopez17RNN-like network for tests with HEADER input (n_inputs, n_features)."""

    def __init__(self, in_channels=1, num_classes=10, **kwargs):
        super().__init__()

        num_fields = kwargs['num_fields']
        out_features_size = kwargs['out_features_size']
        if out_features_size is None:
            out_features_size = 100

        # main part of the network
        # nn.LSTM takes in input:
        # -input_size: H_in
        # -hidden_size: H_cell == H_out if proj_size=0 (default)
        # -batch_first=True, to pass input of shape (N, L, H_in).
        #  N is batch size and L the sequence length (num_pkts).
        # and will output:
        # (N, L, D * H_out), where D is 2 if bidirectional, else 1 (default).
        scaling_factor = kwargs.get('scale', 1)
        hidden_size = max([math.ceil(100 * scaling_factor), 50])

        self.rnn = nn.LSTM(num_fields, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, out_features_size)

        # last classifier layer (head) with as many outputs as classes
        self.fc = nn.Linear(out_features_size, num_classes)
        # and `head_var` with the name of the head, so it can be removed when doing incremental learning experiments
        self.head_var = 'fc'

    def forward(self, x):
        out = F.relu(self.extract_features(x))
        out = self.fc(out)
        return out

    def extract_features(self, x):
        in_size = x.size()
        out = torch.reshape(x, (in_size[0], in_size[2], in_size[3]))
        # In order to reproduce the behavior of keras LSTM, we take the last sequence only and apply the tanh activation
        self.rnn.flatten_parameters()
        out, _ = self.rnn(out)
        out = torch.tanh(out[:, -1, :])
        out = self.fc1(out)
        return out
