import numpy as np
import torch
from torch import nn
from learn2learn.utils import accuracy
from learn2learn.nn import PrototypicalClassifier

from fsl.approach.meta_module import LightningMetaModule


class LightningPrototypicalNetworks(LightningMetaModule):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/algorithms/lightning/lightning_protonet.py)

    **Description**

    A PyTorch Lightning module for Prototypical Networks.

    **Arguments**

    * **features** (Module) - Feature extractor which classifies input tasks.
    * **loss** (Function, *optional*, default=CrossEntropyLoss) - Loss function which maps the cost of the events.
    * **distance_metric** (str, *optional*, default='euclidean') - Distance metric between samples. ['euclidean', 'cosine']
    * **train_ways** (int, *optional*, default=5) - Number of classes in for train tasks.
    * **train_shots** (int, *optional*, default=1) - Number of support samples for train tasks.
    * **train_queries** (int, *optional*, default=1) - Number of query samples for train tasks.
    * **test_ways** (int, *optional*, default=5) - Number of classes in for test tasks.
    * **test_shots** (int, *optional*, default=1) - Number of support samples for test tasks.
    * **test_queries** (int, *optional*, default=1) - Number of query samples for test tasks.
    * **lr** (float, *optional*, default=0.001) - Learning rate of meta training.
    * **scheduler_patience** (int, *optional*, default=20) - Patience for `lr`.
    * **scheduler_decay** (float, *optional*, default=1.0) - Decay rate for `lr`.

    **References**

    1. Snell et al. 2017. "Prototypical Networks for Few-shot Learning"

    **Example**

    ~~~python
    tasksets = l2l.vision.benchmarks.get_tasksets('mini-imagenet')
    features = Convnet()  # init model
    protonet = LightningPrototypicalNetworks(features, **dict_args)
    episodic_data = EpisodicBatcher(tasksets.train, tasksets.validation, tasksets.test)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(protonet, episodic_data)
    ~~~
    """

    distance_metric = "euclidean"

    def __init__(self, net, loss=None, **kwargs):
        super(LightningPrototypicalNetworks, self).__init__(**kwargs)
        
        # Meta-Learning specific parameters
        self.loss = loss or nn.CrossEntropyLoss(reduction="mean")
        self.distance_metric = kwargs.get(
            "distance_metric", LightningPrototypicalNetworks.distance_metric)
        self.net = net
        self.classifier = PrototypicalClassifier(distance=self.distance_metric)
        
        self.save_hyperparameters({
            "train_ways": self.train_ways,
            "train_shots": self.train_shots,
            "train_queries": self.train_queries,
            "test_ways": self.test_ways,
            "test_shots": self.test_shots,
            "test_queries": self.test_queries,
            "lr": self.lr,
            "lr_strat": self.lr_strat,
            "distance_metric": self.distance_metric,
            "loss_factor": self.loss_factor,
            "add_cl": self.add_cl,
        })
          
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = LightningMetaModule.add_model_specific_args(parent_parser)
        parser.add_argument(
            "--distance_metric",
            type=str,
            default=LightningPrototypicalNetworks.distance_metric,
        )
        return parser


    def meta_learn(self, batch, batch_idx, N, K_s, K_q):
        self.net.train()
        data, labels = batch
        labels, le = self.label_encoding(labels)

        # Sort data samples by labels
        sort = torch.sort(labels)
        data = data.squeeze(0)[sort.indices].squeeze(0)
        labels = labels.squeeze(0)[sort.indices].squeeze(0)
        # Compute support and query embeddings
        support_indices = np.zeros(data.size(0), dtype=bool)
        selection = np.arange(N) * (K_s + K_q)
        for offset in range(K_s):
            support_indices[selection + offset] = True

        query_indices = torch.from_numpy(~support_indices)
        support_indices = torch.from_numpy(support_indices)

        # Adapt to multimodal input if net_is_mm
        data = self.split_multimodal(data, data.size(0))

        _, embeddings = self.net(data, return_features=True, stage=self.stage)
        support = embeddings[support_indices]
        support_labels = labels[support_indices]
        query = embeddings[query_indices]
        query_labels = labels[query_indices]

        self.classifier.fit_(support, support_labels)
        logits = self.classifier(query)
        eval_loss = self.loss(logits, query_labels)
        eval_accuracy = accuracy(logits, query_labels)
        
        return {
            'loss': eval_loss,
            'accuracy': eval_accuracy,
            'logits': logits,
            'query_labels': query_labels,
            'support_labels': support_labels,
            'le_map': le,
            'support': support,
            'query': query
        }
