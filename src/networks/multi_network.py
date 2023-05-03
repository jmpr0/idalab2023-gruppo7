from copy import deepcopy

import numpy as np
import torch
from torch import nn

from networks.shallow_autoencoder import ShallowAutoencoder

from .network import LLL_Net


class LLL_MultiNet(nn.Module):
    """Basic class for implementing networks"""

    def __init__(self, model, modality='all', remove_existing_head=None, activate_features=True):
        head_var = model.head_var
        assert type(head_var) == str
        assert not remove_existing_head or hasattr(model, head_var), \
            "Given model does not have a variable called {}".format(head_var)
        assert not remove_existing_head or type(getattr(model, head_var)) in [nn.Sequential, nn.Linear], \
            "Given model's head {} does is not an instance of nn.Sequential or nn.Linear".format(head_var)
        super(LLL_MultiNet, self).__init__()

        self.base_model = model
        self.modality = modality
        self.activate_features = activate_features

        self.model = nn.ModuleList()

        self.task_cls_list = []
        self.task_cls = None
        self.task_offset = []

    def add_head(self, num_outputs, binary=True, **kwargs):
        """Add a new head with the corresponding number of outputs. Also update the number of classes per task and the
        corresponding offsets
        """
        if isinstance(self.base_model, ShallowAutoencoder):
            print("Shallow Autoencoder")
            net = ShallowAutoencoder()
            self.model.append(net)
        elif binary:
            for _ in range(num_outputs): # add one single_net for each class
                net = LLL_Net(deepcopy(self.base_model), remove_existing_head=True, modality=self.modality,
                                 activate_features=self.activate_features)
                net.add_head(1)
                self.model.append(net)
        else:
            net = LLL_Net(deepcopy(self.base_model), remove_existing_head=True, modality=self.modality,
                                activate_features=self.activate_features)
            net.add_head(num_outputs)
            self.model.append(net)

        self.task_cls_list.append(torch.tensor([num_outputs]))
        self.task_cls = torch.cat(self.task_cls_list)
        self.task_offset = torch.cat([torch.LongTensor(1).zero_(), self.task_cls.cumsum(0)[:-1]])
        print(net)

    def forward(self, x, return_features=False):
        """Applies the forward pass

        Simplification to work on multi-head only -- returns all head outputs in a list
        Args:
            x (tensor): input images
            return_features (bool): return the representations before the heads
        """
        rets = []
        for model in self.model:
            rets.append(model.forward(x, return_features))
        if return_features:
            return [*zip(*rets)]
        else:
            return rets

    def get_copy(self):
        """Get weights from the model"""
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        """Load weights into the model"""
        self.load_state_dict(deepcopy(state_dict))
        return

    def unfreeze_all(self, t=0, freezing=None, verbose=True):
        """Unfreeze all parameters from the model, including the heads"""
        for name, param in self.model[t].named_parameters():
            if not freezing or not sum(nf in name for nf in freezing):
                param.requires_grad = True
        if verbose:
            self.trainability_info()

    def freeze_all(self, t=0, non_freezing=None, verbose=True):
        """Freeze all parameters from the model, including the heads"""
        for name, param in self.model[t].named_parameters():
            if not non_freezing or not sum(nf in name for nf in non_freezing):
                param.requires_grad = False
        if verbose:
            self.trainability_info()

    def freeze_backbone(self, t=0, non_freezing=None, verbose=True):
        """Freeze all parameters from the main model, but not the heads"""
        for name, param in self.model[t].model.named_parameters():
            if not non_freezing or not sum(nf in name for nf in non_freezing):
                param.requires_grad = False
        if verbose:
            self.trainability_info()
        
    def trainability_info(self):
        print('\nTRAINABILITY INFO')
        for name, param in self.named_parameters():
            print(name, param.requires_grad)
        print('')

    def freeze_bn(self, t=0):
        """Freeze all Batch Normalization layers from the model and use them in eval() mode"""
        for m in self.model[t].modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _initialize_weights(self):
        """Initialize weights using different strategies"""
        # TODO: add different initialization strategies
        pass
