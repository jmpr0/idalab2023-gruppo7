import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math

from .network import get_output_dim


class Lopez17RNNCNN(nn.Module):
    """Lopez17RNNCNN-like network for tests with HEADER input (n_inputs, n_features)."""

    def __init__(self, in_channels=1, num_classes=10, **kwargs):
        super().__init__()

        num_fields = kwargs['num_fields']
        out_features_size = kwargs['out_features_size']
        if out_features_size is None:
            out_features_size = 100

        rnn_hidden_size = 100

        kernel = 25
        stride = 1
        self.pool_kernel = 3
        self.pool_stride = self.pool_kernel

        features_size, self.paddings = get_output_dim(
            rnn_hidden_size,
            kernels=[kernel, self.pool_kernel, kernel, self.pool_kernel],
            strides=[stride, self.pool_stride, stride, self.pool_stride],
            padding='same',
            return_paddings=True
        )

        # main part of the network
        self.rnn = nn.LSTM(num_fields, rnn_hidden_size, batch_first=True)

        scaling_factor = kwargs.get('scale', 1)
        filters = [math.ceil(32 * scaling_factor), math.ceil(64 * scaling_factor)]

        self.conv1 = nn.Conv1d(in_channels, filters[0], kernel, stride=stride, padding=0)
        self.bn1 = nn.BatchNorm1d(filters[0])
        self.conv2 = nn.Conv1d(filters[0], filters[1], kernel, stride=stride, padding=0)
        self.bn2 = nn.BatchNorm1d(filters[1])
        self.fc1 = nn.Linear(features_size * filters[1], out_features_size)

        # last classifier layer (head) with as many outputs as classes
        self.fc = nn.Linear(out_features_size, num_classes)
        # and `head_var` with the name of the head, so it can be removed when doing incremental learning experiments
        self.head_var = 'fc'

    def forward(self, x):
        out = F.relu(self.extract_features(x))
        out = F.dropout(out, 0.4)
        out = self.fc(out)
        return out

    def extract_features(self, x):
        in_size = x.size()
        out = torch.reshape(x, (in_size[0], in_size[2], in_size[3]))
        # Because the LSTM output will fed the CNN, we take its entire output activated by tanh
        self.rnn.flatten_parameters()
        out, _ = self.rnn(out)
        # out = torch.tanh(out)
        out = torch.tanh(out[:, -1, :])
        out = F.dropout(out, 0.2)
        # We force the channel size of the LSTM output (viz. CNN input) to be 1
        size_interm = out.size()

        # out = torch.reshape(out, (size_interm[0], 1, size_interm[1], size_interm[2]))
        # out = F.pad(out, self.paddings0[0] + self.paddings1[0])
        # out = F.relu(self.conv1(out))
        # out = self.bn1(out)
        # out = F.pad(out, self.paddings0[1] + self.paddings1[1])
        # out = F.relu(self.conv2(out))
        # out = self.bn2(out)

        out = torch.reshape(out, (size_interm[0], 1, size_interm[1]))
        out = F.pad(out, self.paddings[0])
        out = F.relu(self.conv1(out))
        out = self.bn1(out)
        out = F.pad(out, self.paddings[1])
        out = F.max_pool1d(out, self.pool_kernel, stride=self.pool_stride, padding=0)
        out = F.pad(out, self.paddings[2])
        out = F.relu(self.conv2(out))
        out = self.bn2(out)
        out = F.pad(out, self.paddings[3])
        out = F.max_pool1d(out, self.pool_kernel, stride=self.pool_stride, padding=0)

        out = torch.flatten(out, start_dim=1)
        out = self.fc1(out)
        return out
