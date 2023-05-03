import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .aceto19cnn import Aceto19CNN
from .lopez17cnn import Lopez17CNN
from .network import get_output_dim


class Aceto19MIMETICP(nn.Module):
    """MIMETIC+ multimodal model."""

    def __init__(self, in_channels=1, num_classes=10, **kwargs):
        super().__init__()

        out_features_size = kwargs['out_features_size']
        if out_features_size is None:
            out_features_size = 128
        #TODO: add scale
        kwargs0 = dict(
            num_pkts=None, num_fields=None, num_bytes=kwargs['num_bytes'],
            out_features_size=kwargs['out_features_size'])
        kwargs1 = dict(
            num_pkts=kwargs['num_pkts'], num_fields=kwargs['num_fields'], num_bytes=None,
            out_features_size=kwargs['out_features_size'])

        # Modality-specific models
        self.modspec_models = nn.ModuleList([
            Aceto19CNN(in_channels=in_channels, **kwargs0),
            Lopez17CNN(in_channels=in_channels, **kwargs1)
        ])

        modspec_features_sizes = [getattr(modspec_model, modspec_model.head_var).in_features 
                                  for modspec_model in self.modspec_models]

        self.fc1 = nn.Linear(sum(modspec_features_sizes), out_features_size)

        # last classifier layer (head) with as many outputs as classes
        self.fc = nn.Linear(out_features_size, num_classes)
        # and `head_var` with the name of the head, so it can be removed when doing incremental learning experiments
        self.head_var = 'fc'

        self.activation = F.relu
        # self.activation = torch.tanh

    def forward(self, xs):
        out = self.activation(self.extract_features(xs))
        if self.training:  # The dropout still works in eval mode. We explicitely avoid it's usage.
            out = F.dropout(out, 0.2)
        out = self.fc(out)
        return out

    def extract_features(self, xs):
#         print('\nMIMETIC\n')
#         print([id(modspec_model) for modspec_model in self.modspec_models])
#         print([type(modspec_model) for modspec_model in self.modspec_models])
#         print(['%.2f' % np.sum(l.cpu().numpy()) for l in list(self.modspec_models[0].state_dict().values())])
#         print(['%.2f' % np.sum(l.cpu().numpy()) for l in list(self.modspec_models[1].state_dict().values())])
#         input()
#         from time import sleep
#         sleep(.1)
        outs = [modspec_model.extract_features(x) for modspec_model, x in zip(self.modspec_models, xs)]
        out = self.activation(torch.cat(outs, 1))
        if self.training:  # The dropout still works in eval mode. We explicitely avoid it's usage.
            out = F.dropout(out, 0.2)
        out = self.fc1(out)
        return out
