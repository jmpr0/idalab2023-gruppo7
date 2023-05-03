import numpy as np
import torch
import learn2learn as l2l
from learn2learn.utils import accuracy
from torch import nn

from fsl.approach.meta_module import LightningMetaModule
from fsl.approach.maml import LightningMAML


class LightningANIL(LightningMetaModule):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/algorithms/lightning/lightning_anil.py)

    **Description**

    A PyTorch Lightning module for ANIL.

    **Arguments**

    * **features** (Module) - A nn.Module to extract features, which will not be adaptated.
    * **classifier** (Module) - A nn.Module taking features, mapping them to classification.
    * **loss** (Function, *optional*, default=CrossEntropyLoss) - Loss function which maps the cost of the events.
    * **ways** (int, *optional*, default=5) - Number of classes in a task.
    * **shots** (int, *optional*, default=1) - Number of samples for adaptation.
    * **adaptation_steps** (int, *optional*, default=1) - Number of steps for adapting to new task.
    * **lr** (float, *optional*, default=0.001) - Learning rate of meta training.
    * **adaptation_lr** (float, *optional*, default=0.1) - Learning rate for fast adaptation.
    * **scheduler_patience** (int, *optional*, default=20) - Patience for `lr`.
    * **scheduler_decay** (float, *optional*, default=1.0) - Decay rate for `lr`.

    **References**

    1. Raghu et al. 2020. "Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness of MAML"

    **Example**

    ~~~python
    tasksets = l2l.vision.benchmarks.get_tasksets('omniglot')
    model = l2l.vision.models.OmniglotFC(28**2, args.ways)
    anil = LightningANIL(model.features, model.classifier, adaptation_lr=0.1, **dict_args)
    episodic_data = EpisodicBatcher(tasksets.train, tasksets.validation, tasksets.test)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(anil, episodic_data)
    ~~~
    """

    def __init__(self, net, loss=None, **kwargs):
        super(LightningANIL, self).__init__(**kwargs)
        
        # Meta-Learning specific parameters
        self.loss = loss or nn.CrossEntropyLoss(reduction="mean")
        self.adaptation_steps = kwargs.get("adaptation_steps", LightningMAML.adaptation_steps)
        self.adaptation_lr = kwargs.get("adaptation_lr", LightningMAML.adaptation_lr)
        self.net = net
        # Add stubs (for multi-modal nets) & model head to the classifiers  
        self.classifiers = nn.ModuleList() 
        if hasattr(net, 'stubs'):
            for stub in net.stubs:
                self.classifiers.append(l2l.algorithms.MAML(stub, lr=self.adaptation_lr))
        self.classifiers.append(l2l.algorithms.MAML(net.heads[0], lr=self.adaptation_lr))
        
        assert (
            self.train_ways == self.test_ways
        ), "For ANIL, train_ways should be equal to test_ways."
        
        self.save_hyperparameters({
            "train_ways": self.train_ways,
            "train_shots": self.train_shots,
            "train_queries": self.train_queries,
            "test_ways": self.test_ways,
            "test_shots": self.test_shots,
            "test_queries": self.test_queries,
            "lr": self.lr,
            "lr_strat": self.lr_strat,
            "adaptation_lr": self.adaptation_lr,
            "adaptation_steps": self.adaptation_steps,
            "loss_factor": self.loss_factor,
            "add_cl": self.add_cl,
        })


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = LightningMetaModule.add_model_specific_args(parent_parser)
        parser.add_argument(
            "--adaptation_steps",
            type=int,
            default=LightningMAML.adaptation_steps,
        )
        parser.add_argument(
            "--adaptation_lr",
            type=float,
            default=LightningMAML.adaptation_lr,
        )
        return parser

    @torch.enable_grad()
    def meta_learn(self, batch, batch_idx, N, K_s, K_q):
        self.net.train()
        learner = self.classifiers[self.stage].clone()
        learner.train()
        data, labels = batch
        labels, le = self.label_encoding(labels)

        # Separate data into adaptation and evaluation sets
        support_indices = np.zeros(data.size(0), dtype=bool)
        selection = np.arange(N) * (K_s + K_q)
        for offset in range(K_s):
            support_indices[selection + offset] = True

        query_indices = torch.from_numpy(~support_indices)
        support_indices = torch.from_numpy(support_indices)

        # Adapt to multimodal input if net_is_mm
        data = self.split_multimodal(data, data.size(0))

        _, data = self.net(data, return_features=True, stage=self.stage)
        support = data[support_indices]
        support_labels = labels[support_indices]
        query = data[query_indices]
        query_labels = labels[query_indices]

        # Adapt the classifier
        for step in range(self.adaptation_steps):
            preds = learner(support)
            train_error = self.loss(preds, support_labels)
            learner.adapt(train_error)

        # Evaluating the adapted model
        predictions = learner(query)
        valid_error = self.loss(predictions, query_labels)
        valid_accuracy = accuracy(predictions, query_labels)

        return {
            'loss': valid_error,
            'accuracy': valid_accuracy,
            'logits': predictions,
            'query_labels': query_labels,
            'support_labels': support_labels,
            'le_map': le,
            'support': support,
            'query': query
        }
