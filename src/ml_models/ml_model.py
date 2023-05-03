from copy import deepcopy
import numpy as np
import inspect


class ML_Model():
    """Basic class for implementing ml models"""

    def __init__(self, model, **kwargs):
        self.model = model
        
        self.task_cls = []
        self.task_offset = []
        
    def add_head(self, num_outputs, **kwargs):
        """Add a new head with the corresponding number of outputs. Also update the number of classes per task and the
        corresponding offsets
        """
        attrs = inspect.signature(self.model.__class__.__init__).parameters.keys()
        self.model = self.model.__class__(**dict([(attr, self.model.model.__dict__[attr]) for attr in attrs if attr != 'self']))
        
        self.task_cls = np.concatenate((self.task_cls, [num_outputs]))
        self.task_offset = np.concatenate(([0], self.task_cls.cumsum(0)[:-1]))
        
    def fit(self, x, y):
        return self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict_proba(x)
    
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
