import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Aceto19RNN(nn.Module):
    """Lopez17RNN-like network used as a single modality by Aceto19 for tests with HEADER input."""

    def __init__(self, in_channels=1, num_classes=10, **kwargs):
        super().__init__()

        num_fields = kwargs['num_fields']
        num_pkts = kwargs['num_pkts']
        out_features_size = kwargs['out_features_size']
        if out_features_size is None:
            out_features_size = 256
        # TODO: implement variational dropout
        # self.variational_dropout = kwargs.get('variational_dropout', False)
        # self.var_dropout_factor = kwargs.get('var_dropout_factor', 100)

        scaling_factor = kwargs.get('scale', 1)
        filter = math.ceil(64 * scaling_factor)

        # main part of the network
        # nn.GRU takes in input:
        # -input_size: H_in
        # -hidden_size: H_out
        # -batch_first=True, to pass input of shape (N, L, H_in).
        #  N is batch size and L the sequence length (num_input[0]).
        # and will output:
        # (N, L, D * H_out), where D is 2 if bidirectional, else 1 (default).
        self.rnn = nn.GRU(num_fields, filter, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(num_pkts * 2 * filter, out_features_size)  # L * D * H_out as input size

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
        in_size = x.size()
        out = torch.reshape(x, (in_size[0], in_size[2], in_size[3]))
        # In the MIMETIC tf implementation, the entire BiGRU output is taken, then activated via ReLU, and flattened
        self.rnn.flatten_parameters()
        out, _ = self.rnn(out)
        out = self.activation(out)
        out = torch.flatten(out, start_dim=1)
        if self.training:  # The dropout still works in eval mode. We explicitely avoid it's usage.
            out = F.dropout(out, 0.2)
        out = self.fc1(out)
        return out
