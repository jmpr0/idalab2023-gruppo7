import numpy as np
from torch import nn
import torch.nn.functional as F


class RecurrentAutoencoder(nn.Module):
    """doc string"""
    
    def __init__(self, num_classes=10, **kwargs):
        super().__init__()
        
        embedding_dim = 64
        out_features_size = embedding_dim * kwargs['num_pkts']
        self.encoder = EncoderRNN(
            kwargs['num_pkts'], kwargs['num_fields'], embedding_dim
        )
        self.decoder = DecoderRNN(
            kwargs['num_pkts'], embedding_dim, kwargs['num_fields']
        )
        
        # last classifier layer (head) with as many outputs as classes
        self.fc = nn.Linear(out_features_size, num_classes)
        # and `head_var` with the name of the head, so it can be removed when doing incremental learning experiments
        self.head_var = 'fc'
        
    def forward(self, x):
        out, _ = F.relu(self.extract_features(x))
        out = self.fc(out)
    
    def extract_features(self, x):
        out, hidden = self.encoder(x) 
        return out.flatten(start_dim=1), hidden
    
    def extract_reconstruction(self, x):
        out = self.decoder(x)
        return out
    

class EncoderRNN(nn.Module):

  def __init__(self, seq_len, n_features, embedding_dim=64):
    super().__init__()
    
    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

    self.rnn1 = nn.LSTM(
        input_size=n_features,
        hidden_size=self.hidden_dim,
        num_layers=1,
        batch_first=True
    )
    
    self.rnn2 = nn.LSTM(
        input_size=self.hidden_dim,
        hidden_size=embedding_dim,
        num_layers=1,
        batch_first=True
    )

  def forward(self, x):
    in_size = x.size()
    x = x.reshape((in_size[0], in_size[2], in_size[3]))

    x, (_, _) = self.rnn1(x)
    x, (hidden, _) = self.rnn2(x)

    return x, hidden.squeeze(dim=0)

class DecoderRNN(nn.Module):

  def __init__(self, seq_len, input_dim=64, n_features=1):
    super().__init__()

    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = 2 * input_dim, n_features

    self.rnn1 = nn.LSTM(
        input_size=input_dim,
        hidden_size=input_dim,
        num_layers=1,
        batch_first=True
    )

    self.rnn2 = nn.LSTM(
        input_size=input_dim,
        hidden_size=self.hidden_dim,
        num_layers=1,
        batch_first=True
    )
    
    self.output_layer = nn.Linear(self.hidden_dim, n_features)

  def forward(self, x):
    in_size = x.size()

    x = x.repeat(self.seq_len, 1)
    x = x.reshape((in_size[0], self.seq_len, self.input_dim))

    x, (_, _) = self.rnn1(x) # x, (hidden_n, cell_n)
    x, (_, _) = self.rnn2(x)
    x = x.reshape((in_size[0], self.seq_len, self.hidden_dim))
    
    return self.output_layer(x)