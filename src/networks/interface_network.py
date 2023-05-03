import importlib
from torch import nn


class INetwork(nn.Module):
    
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def factory_net(model, remove_existing_head=False, modality='all',
                activate_features=True, weights_path=None):
        """
        Return a MultiModalNet (for multi-modal net) or a LLL_Net (for single-modal net)
        and the number of modalities
        """
        if hasattr(model, 'modspec_models'):
            Network = getattr(importlib.import_module(
                name='networks.multimodal_network'), 'MultiModalNet')
            is_mm = True
        else:
            Network = getattr(importlib.import_module(
                name='networks.network'), 'LLL_Net')
            is_mm = False
        
        net = Network(model, remove_existing_head, modality, activate_features, weights_path)
        num_modalities = len(net.modalities) if is_mm else 0
        print(net)
        return net, num_modalities