import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math

from .network import get_output_dim


class Aceto19CNN(nn.Module):
    """Wang17-like network used as a single modality by Aceto19 for tests with PAYLOAD input."""

    def __init__(self, in_channels=1, num_classes=10, **kwargs):
        super().__init__()

        seq_length = kwargs['num_bytes'] or kwargs['num_pkts']
        in_channels = kwargs['num_fields'] or in_channels
        out_features_size = kwargs['out_features_size']
        if out_features_size is None:
            out_features_size = 256

        kernel = 25
        stride = 1
        self.pool_kernel = 3
        self.pool_stride = self.pool_kernel

        for padding in ['valid', 'same']:
            features_size, self.paddings = get_output_dim(
                seq_length,
                kernels=[kernel, self.pool_kernel, kernel, self.pool_kernel],
                strides=[stride, self.pool_stride, stride, self.pool_stride],
                padding=padding,
                return_paddings=True
            )
            if features_size < 1:  # Apply padding 'same' at each layer
                pass
            else:
                print(padding)
                break

        scaling_factor = kwargs.get('scale', 1)
        filters = [math.ceil(16 * scaling_factor), math.ceil(32 * scaling_factor)]

        # main part of the network
        self.conv1 = nn.Conv1d(in_channels, filters[0], kernel, stride=stride, padding=0)
        self.conv2 = nn.Conv1d(filters[0], filters[1], kernel, stride=stride, padding=0)
        self.fc1 = nn.Linear(features_size * filters[1], out_features_size)

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
        # When one pass the Lopez input, features are treated as chennels.
        if len(x.size()) == 4:
            x = x.squeeze(dim=1).permute(0, 2, 1)
        out = F.pad(x, self.paddings[0])
        out = self.activation(self.conv1(out))
        out = F.pad(out, self.paddings[1])
        out = F.max_pool1d(out, self.pool_kernel, stride=self.pool_stride, padding=0)
        out = F.pad(out, self.paddings[2])
        out = self.activation(self.conv2(out))
        out = F.pad(out, self.paddings[3])
        out = F.max_pool1d(out, self.pool_kernel, stride=self.pool_stride, padding=0)
        out = torch.flatten(out, start_dim=1)
        # TODO: out.requires_grad
        if self.training:  # The dropout still works in eval mode. We explicitely avoid it's usage.
            out = F.dropout(out, 0.2)
        out = self.fc1(out)
        return out