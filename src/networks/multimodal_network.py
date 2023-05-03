from torch import nn
import torch

from networks.interface_network import INetwork

class MultiModalNet(INetwork):  

    def __init__(self, model, remove_existing_head=False, modality='all',
                 activate_features=True, weights_path=None):
        super().__init__()
        
        head_var = model.head_var
        assert type(head_var) == str
        assert not remove_existing_head or hasattr(model, head_var), \
            "Given model does not have a variable called {}".format(head_var)
        assert not remove_existing_head or type(getattr(model, head_var)) in [nn.Sequential, nn.Linear], \
            "Given model's head {} does is not an instance of nn.Sequential or nn.Linear".format(head_var)

        self.model = model
        self.modality = modality
        
        if activate_features:
            self.feat_activation = nn.functional.relu
        else:
            self.feat_activation = nn.Identity()
            
        last_layer = getattr(self.model, head_var)

        if remove_existing_head:
            if type(last_layer) == nn.Sequential:
                print('1')
                self.out_size = last_layer[-1].in_features
                del last_layer[-1]
            elif type(last_layer) == nn.Linear:
                print('2')
                self.out_size = last_layer.in_features
                setattr(self.model, head_var, nn.Sequential())
        else:
            self.out_size = last_layer.out_features
        
        self.heads = nn.ModuleList()
        self.stubs = nn.ModuleList()
        self.modalities = list(range(len(self.model.modspec_models))) # List of int for each mod
        self._initialize_weights(weights_path)
        
    def add_head(self, num_outputs, **kwargs):
        """
        Add a new head with the corresponding number of outputs.
        """
        last_layer_size = self.out_size if not kwargs['concat'] else 640 # TODO automatizzare
        self.heads.append(nn.Linear(last_layer_size, num_outputs))
        
        # For each modality create a stub 
        for mod in self.modalities:
            last_mod_layer = getattr(self.model.modspec_models[mod], 'fc')
            self.stubs.append(nn.Linear(last_mod_layer.in_features, num_outputs))
        self.model.return_concat_features = kwargs['concat']
        
    def forward(self, x, return_features=False, **kwargs):
        y = []
        stage = kwargs.get('stage', 0)
        if stage in self.modalities:
            # Sigle-modality pre-training
            x = self.model.modspec_models[stage].extract_features(x[stage])
            y.append(self.stubs[stage](x))
        elif stage == len(self.modalities):
            # Shared layer fine-tuning
            x = self.model.extract_features(x)
            x = x[-1] if isinstance(x, list) else x
            assert (len(self.heads) > 0), "Cannot access any head"
            for head in self.heads:
                y.append(head(self.feat_activation(x)))
        else:
            raise ValueError("Invalid stage")
        if return_features:            
            return y, x
        else:
            return y
          
    def freeze_mod_backbone(self): 
        # Frezes the backbones of single-modality branches with the exception of the last layer
        for mod in self.modalities: 
            for name, param in self.model.modspec_models[mod].named_parameters():
                if name not in ['fc1.weight', 'fc1.bias']:
                    param.requires_grad = False
                    
    def _initialize_weights(self, path):
        """Initialize weights using different strategies"""
        if path is None:
            return
        self.model.load_state_dict(torch.load(path))
        print(f'Model loaded from {path}')