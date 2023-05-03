import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Chen21RNN(nn.Module):
    """Chen21RNN network by chen2021 with PLs input."""

    def __init__(self, in_channels=1, num_classes=10, **kwargs):
        super().__init__()

        num_pkts = kwargs['num_pkts']
        out_features_size = kwargs['out_features_size']
        if out_features_size is None:
            out_features_size = 32

        # main part of the network
        self.embedding = nn.Embedding(num_embeddings=1461, embedding_dim=128, padding_idx=0)
        # nn.GRU takes in input:
        # -input_size: H_in
        # -hidden_size: H_out
        # -batch_first=True, to pass input of shape (N, L, H_in).
        #  N is batch size and L the sequence length (num_input[0]).
        # and will output:
        # (N, L, D * H_out), where D is 2 if bidirectional, else 1 (default).
        self.rnn = nn.GRU(128, 128, batch_first=True)
        self.fc2 = nn.Linear(128, 128)  # L * D * H_out as input size
        self.fc1 = nn.Linear(128, out_features_size)

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
        out = (x * 1460).int()  # TODO: works only with PL field
        in_size = out.size()
        out = torch.reshape(out, (in_size[0], in_size[2]))
        out = self.embedding(out)
        # In order to reproduce the behavior of keras LSTM, we take the last sequence only and apply the tanh activation
        self.rnn.flatten_parameters()
        out, _ = self.rnn(out)
        out = self.activation(out[:, -1, :])
        out = F.dropout(out, 0.3)
        out = self.fc2(out)
        out = self.activation(out)
        out = self.fc1(out)
        return out
