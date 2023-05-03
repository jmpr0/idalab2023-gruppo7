from copy import deepcopy

import torch
from torch import nn

from networks.interface_network import INetwork


class LLL_Net(INetwork):
    """Basic class for implementing networks"""

    def __init__(self, model, remove_existing_head=False, modality='all',
                 activate_features=True, weights_path=None, ae_pretrain=False):
        head_var = model.head_var
        assert type(head_var) == str
        assert not remove_existing_head or hasattr(model, head_var), \
            "Given model does not have a variable called {}".format(head_var)
        assert not remove_existing_head or type(getattr(model, head_var)) in [nn.Sequential, nn.Linear], \
            "Given model's head {} does is not an instance of nn.Sequential or nn.Linear".format(head_var)
        super(LLL_Net, self).__init__()

        self.model = model
        self.modality = modality

        if activate_features:
            self.feat_activation = nn.functional.relu
        else:
            self.feat_activation = nn.Identity()

        last_layer = getattr(self.model, head_var)

        if remove_existing_head:
            if type(last_layer) == nn.Sequential:
                self.out_size = last_layer[-1].in_features
                # strips off last linear layer of classifier
                del last_layer[-1]
            elif type(last_layer) == nn.Linear:
                self.out_size = last_layer.in_features
                # converts last layer into identity
                # setattr(self.model, head_var, nn.Identity())
                # WARNING: this is for when pytorch version is <1.2
                setattr(self.model, head_var, nn.Sequential())
        else:
            self.out_size = last_layer.out_features

        self.heads = nn.ModuleList()
        self.task_cls = []
        self.task_offset = []
        self.return_reconstruction = hasattr(self.model, 'decoder') and ae_pretrain
        self._initialize_weights(weights_path)

    def add_head(self, num_outputs, **kwargs):
        """Add a new head with the corresponding number of outputs. Also update the number of classes per task and the
        corresponding offsets
        """
        last_layer_size = self.out_size if not kwargs.get('concat', False) else 640 # TODO automatizzare il 640
        self.heads.append(nn.Linear(last_layer_size, num_outputs))
        # we re-compute instead of append in case an approach makes changes to the heads
        self.task_cls = torch.tensor([head.out_features for head in self.heads])
        self.task_offset = torch.cat([torch.LongTensor(1).zero_(), self.task_cls.cumsum(0)[:-1]])

    def forward(self, x, return_features=False, **kwargs):
        """Applies the forward pass

        Simplification to work on multi-head only -- returns all head outputs in a list
        Args:
            x (tensor): input images
            return_features (bool): return the representations before the heads
        """
        # TODO: using extract_features is not compliant w/ computer vision models. Think to an alternative.
        try:
            x = self.model.extract_features(x[self.modality])
        except TypeError as _:
            x = self.model.extract_features(x)
        assert (len(self.heads) > 0), "Cannot access any head"
        y = []
        for head in self.heads:
            y.append(head(self.feat_activation(x)))
        if self.return_reconstruction:
            x_bar = self.model.extract_reconstruction(x)
            return y, x, x_bar
        elif return_features:
            return y, x
        else:
            return y

    def get_copy(self):
        """Get weights from the model"""
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        """Load weights into the model"""
        self.load_state_dict(deepcopy(state_dict))
        return

    def unfreeze_all(self, t=0, freezing=None, verbose=True):
        """Unfreeze all parameters from the model, including the heads"""
        for name, param in self.named_parameters():
            if not freezing or not sum(nf in name for nf in freezing):
                param.requires_grad = True
        if verbose:
            self.trainability_info()

    def freeze_all(self, t=0, non_freezing=None, verbose=True):
        """Freeze all parameters from the model, including the heads"""
        for name, param in self.named_parameters():
            if not non_freezing or not sum(nf in name for nf in non_freezing):
                param.requires_grad = False
        if verbose:
            self.trainability_info()

    def freeze_backbone(self, t=0, non_freezing=None, verbose=True):
        """Freeze all parameters from the main model, but not the heads"""
        for name, param in self.model.named_parameters():
            if not non_freezing or not sum(nf in name for nf in non_freezing):
                param.requires_grad = False
        if verbose:
            self.trainability_info()
        
    def trainability_info(self):
        print('\nTRAINABILITY INFO')
        for name, param in self.named_parameters():
            print(name, param.requires_grad)
        print('')

    def freeze_bn(self, t=0, verbose=True):
        """Freeze all Batch Normalization layers from the model and use them in eval() mode"""
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def freeze_heads(self, index):
        for i in index:
            for param in self.heads[i].parameters():
                param.requires_grad = False

    def _initialize_weights(self, path):
        """Initialize weights using different strategies"""
        if path is None:
            return
        self.model.load_state_dict(torch.load(path))
        print(f'Model loaded from {path}')
        

def get_padding(kernel, padding='same'):
    # http://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html
    pad = kernel - 1
    if padding == 'same':
        if kernel % 2:
            return pad // 2, pad // 2
        else:
            return pad // 2, pad // 2 + 1
    return 0, 0


def get_output_dim(dimension, kernels, strides, dilatation=1, padding='same', return_paddings=False):
    # http://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html
    out_dim = dimension
    paddings = []
    if padding == 'same':
        for kernel, stride in zip(kernels, strides):
            paddings.append(get_padding(kernel, padding))
            out_dim = (out_dim + stride - 1) // stride
    else:
        for kernel, stride in zip(kernels, strides):
            paddings.append(get_padding(kernel, padding))
            out_dim = (out_dim - kernel + stride) // stride

    # From pytorch documentation: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    # _out_dim = dimension
    # _paddings = []
    # for kernel, stride in zip(kernels, strides):
    #     _paddings.append(get_padding(kernel, padding))
    #     _out_dim = math.floor((_out_dim + sum(_paddings[-1]) - dilatation * (kernel - 1) - 1) / stride + 1)

    if return_paddings:
        return out_dim, paddings
    return out_dim
