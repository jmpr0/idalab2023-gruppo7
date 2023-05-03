import numpy as np
import torch
from torch import nn
import learn2learn as l2l
from learn2learn.utils import accuracy

from fsl.approach.meta_module import LightningMetaModule


class LightningMAML(LightningMetaModule):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/algorithms/lightning/lightning_maml.py)

    **Description**

    A PyTorch Lightning module for MAML.

    **Arguments**

    * **model** (Module) - A PyTorch nn.Module.
    * **loss** (Function, *optional*, default=CrossEntropyLoss) - Loss function which maps the cost of the events.
    * **ways** (int, *optional*, default=5) - Number of classes in a task.
    * **shots** (int, *optional*, default=1) - Number of samples for adaptation.
    * **adaptation_steps** (int, *optional*, default=1) - Number of steps for adapting to new task.
    * **lr** (float, *optional*, default=0.001) - Learning rate of meta training.
    * **adaptation_lr** (float, *optional*, default=0.1) - Learning rate for fast adaptation.
    * **scheduler_patience** (int, *optional*, default=20) - Patience for `lr`.
    * **scheduler_decay** (float, *optional*, default=1.0) - Decay rate for `lr`.

    **References**

    1. Finn et al. 2017. "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks."

    **Example**

    ~~~python
    tasksets = l2l.vision.benchmarks.get_tasksets('omniglot')
    model = l2l.vision.models.OmniglotFC(28**2, args.ways)
    maml = LightningMAML(classifier, adaptation_lr=0.1, **dict_args)
    episodic_data = EpisodicBatcher(tasksets.train, tasksets.validation, tasksets.test)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(maml, episodic_data)
    ~~~
    """

    adaptation_steps = 1
    adaptation_lr = 0.1
    first_order = False

    def __init__(self, net, loss=None, **kwargs):
        super(LightningMAML, self).__init__(**kwargs)
        
        # Meta-Learning specific parameters
        self.loss = loss or nn.CrossEntropyLoss(reduction="mean")
        self.adaptation_steps = kwargs.get("adaptation_steps", LightningMAML.adaptation_steps)
        self.first_order = kwargs.get("first_order", LightningMAML.first_order)
        self.adaptation_lr = kwargs.get("adaptation_lr", LightningMAML.adaptation_lr)
        self.net = l2l.algorithms.MAML(net, lr=self.adaptation_lr, first_order=self.first_order)
        
        assert (
            self.train_ways == self.test_ways
        ), "For MAML, train_ways should be equal to test_ways." 

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
            "first_order": self.first_order,
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
        parser.add_argument(
            "--first_order",
            action='store_true',
            default=LightningMAML.first_order
        )
        return parser

    @torch.enable_grad()
    def meta_learn(self, batch, batch_idx, N, K_s, K_q):
        learner = self.net.clone()
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
        
        support = data[support_indices]
        support_labels = labels[support_indices]
        query = data[query_indices]
        query_labels = labels[query_indices]

        # Adapt to multimodal input if net_is_mm
        support = self.split_multimodal(support, support.size(0))
        query = self.split_multimodal(query, query.size(0))

        # Adapt the model
        for step in range(self.adaptation_steps): 
            s_pred, s_embeddings = learner(support, return_features=True, stage=self.stage)
            train_error = self.loss(s_pred[0], support_labels)
            learner.adapt(train_error, allow_unused=self.net_is_mm, allow_nograd=self.net_is_mm) 

        # Evaluating the adapted model
        q_pred, q_embeddings = learner(query, return_features=True, stage=self.stage)
        predictions = q_pred[0]
        valid_error = self.loss(predictions, query_labels) 
        valid_accuracy = accuracy(predictions, query_labels)

        return {
            'loss': valid_error,
            'accuracy': valid_accuracy, 
            'logits': predictions, 
            'query_labels': query_labels, 
            'support_labels' : support_labels,
            'le_map': le,
            'support': s_embeddings,
            'query': q_embeddings
        }