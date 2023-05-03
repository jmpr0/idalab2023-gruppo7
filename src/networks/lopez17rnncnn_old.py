import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .network import get_output_dim


class Lopez17RNNCNN(nn.Module):
    """Lopez17RNNCNN-like network for tests with HEADER input (n_inputs, n_features)."""

    def __init__(self, in_channels=1, num_classes=10, **kwargs):
        super().__init__()

        num_pkts = kwargs['num_pkts']
        num_fields = kwargs['num_fields']
        out_features_size = kwargs['out_features_size']
        if out_features_size is None:
            out_features_size = 100

        rnn_hidden_size = 100

        kernel = (4, 2)
        stride = (1, 1)

        # The CNN input size is (batch_size, 1, num_pkts, rnn_hidden_size), where rnn_hidden_size is 100
        features_size0, self.paddings0 = get_output_dim(
            num_pkts,
            kernels=[kernel[0], kernel[0]],
            strides=[stride[0], stride[0]],
            padding='valid',
            return_paddings=True
        )
        features_size1, self.paddings1 = get_output_dim(
            rnn_hidden_size,
            kernels=[kernel[1], kernel[1]],
            strides=[stride[1], stride[1]],
            padding='valid',
            return_paddings=True
        )
        if features_size0 < 1 or features_size1 < 1:
            features_size0, self.paddings0 = get_output_dim(
                num_pkts,
                kernels=[kernel[1], kernel[1]],
                strides=[stride[1], stride[1]],
                padding='same',
                return_paddings=True
            )
            features_size1, self.paddings1 = get_output_dim(
                rnn_hidden_size,
                kernels=[kernel[0], kernel[0]],
                strides=[stride[0], stride[0]],
                padding='same',
                return_paddings=True
            )

        # main part of the network
        self.rnn = nn.LSTM(num_fields, rnn_hidden_size, batch_first=True)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel, stride=stride, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel, stride=stride, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(features_size0 * features_size1 * 64, out_features_size)

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
        out, _ = self.rnn(out)
        out = torch.tanh(out)
        out = F.dropout(out, 0.2)
        # We force the channel size of the LSTM output (viz. CNN input) to be 1
        size_interm = out.size()
        out = torch.reshape(out, (size_interm[0], 1, size_interm[1], size_interm[2]))
        out = F.pad(out, self.paddings0[0] + self.paddings1[0])
        out = F.relu(self.conv1(out))
        out = self.bn1(out)
        out = F.pad(out, self.paddings0[1] + self.paddings1[1])
        out = F.relu(self.conv2(out))
        out = self.bn2(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc1(out)
        return out
