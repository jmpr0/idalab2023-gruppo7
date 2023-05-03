from copy import deepcopy

import numpy as np

from .ml_model import ML_Model


class ML_MultiModel():
    """Basic class for implementing multi ML models"""

    def __init__(self, model, **kwargs):

        self.model = model

        self.models = []

        self.task_cls_list = []
        self.task_cls = None
        self.task_offset = []

    def add_head(self, num_outputs, **kwargs):
        """Add a new head with the corresponding number of outputs. Also update the number of classes per task and the
        corresponding offsets
        """
        model = ML_Model(deepcopy(self.model))
        model.add_head(num_outputs)
        self.models.append(model)

        self.task_cls_list.append([num_outputs])
        self.task_cls = np.concatenate(self.task_cls_list)
        self.task_offset = np.concatenate(([0], self.task_cls.cumsum(0)[:-1]))
        
    def fit(self, t, x, y):
        self.models[t].fit(x, y)

    def predict(self, t, x):
        """Applies the predict pass"""
        rets = []
        for model in self.models:
            rets.append(model.predict(x))
        return rets
    
    def to(self, device=None):
        pass

    def get_copy(self):
        """Get weights from the model"""
        raise NotImplementedError

    def set_state_dict(self, state_dict):
        """Load weights into the model"""
        raise NotImplementedError

    def unfreeze_all(self):
        """Unfreeze all parameters from the model, including the heads"""
        raise NotImplementedError

    def freeze_all(self):
        """Freeze all parameters from the model, including the heads"""
        raise NotImplementedError

    def freeze_backbone(self):
        """Freeze all parameters from the main model, but not the heads"""
        raise NotImplementedError

    def freeze_bn(self):
        """Freeze all Batch Normalization layers from the model and use them in eval() mode"""
        raise NotImplementedError

    def _initialize_weights(self):
        """Initialize weights using different strategies"""
        raise NotImplementedError
