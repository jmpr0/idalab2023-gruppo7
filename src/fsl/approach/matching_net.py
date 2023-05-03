import torch
import numpy as np
from torch import nn
from learn2learn.utils import accuracy

from fsl.approach.meta_module import LightningMetaModule

EPSILON = 1e-8


class LightningMatchingNetworks(LightningMetaModule):
    """
    (adapted from: https://github.com/oscarknagg/few-shot)

    **Description**

    A PyTorch Lightning module for Matching Networks.

    **Arguments**

    * **features** (Module) - Feature extractor which classifies input tasks.
    * **loss** (Function, *optional*, default=CrossEntropyLoss) - Loss function which maps the cost of the events.
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

    1. Vinyals et al. 2017. "Matching Networks for One Shot Learning"
    """

    distance = 'l2'

    def __init__(self, net, loss=None, **kwargs):
        super(LightningMatchingNetworks, self).__init__(**kwargs)
        
        # Meta-Learning specific parameters
        self.loss = loss or nn.NLLLoss()
        self.distance = kwargs.get("distance", LightningMatchingNetworks.distance)
        self.net = net
        self.fce = kwargs.get("fce", None)
        
        self.save_hyperparameters({
            "train_ways": self.train_ways,
            "train_shots": self.train_shots,
            "train_queries": self.train_queries,
            "test_ways": self.test_ways,
            "test_shots": self.test_shots,
            "test_queries": self.test_queries,
            "lr": self.lr,
            "lr_strat": self.lr_strat,
            "distance": self.distance,
            "loss_factor": self.loss_factor,
            "add_cl": self.add_cl,
        })
          
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = LightningMetaModule.add_model_specific_args(parent_parser)
        parser.add_argument(
            "--distance",
            type=str,
            default=LightningMatchingNetworks.distance
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

        if self.fce is not None:
            support, query = self.fce(support, query)

        clipped_y_pred = self.classify(
            query, support, N, K_s, K_q, support_labels)
        eval_loss = self.loss(clipped_y_pred.log(), query_labels)
        eval_accuracy = accuracy(clipped_y_pred, query_labels)

        return {
            'loss': eval_loss,
            'accuracy': eval_accuracy,
            'logits': clipped_y_pred,
            'query_labels': query_labels,
            'support_labels': support_labels,
            'le_map': le,
            'support': support,
            'query': query
        }
        """ if self.stage != 2:
            return out
        mod_quaries = torch.tensor_split(query, [256, 512], dim=1)
        mod_supports = torch.tensor_split(support, [256, 512], dim=1)
        for mod, fn in enumerate(['logits_cnn', 'logits_gru']):
            logits_mod = self.classify(
                mod_quaries[mod], mod_supports[mod], ways, shots, queries, support_labels)
            out[fn] = logits_mod
        return out  """

    def classify(self, query, support, N, K_s, K_q, support_labels):
        distances = pairwise_distances(query, support, self.distance)
        attention = (-distances).softmax(dim=1)
        if attention.shape != (N*K_q, N*K_s):
            raise (ValueError(
                f'Expecting attention Tensor to have shape (n * k_q, n * k_s) = ({N*K_q, N*K_s})'
            ))

        # Classification
        support_labels_onehot = torch.zeros(N*K_s, N, device=self._device)
        support_labels = support_labels.unsqueeze(-1)
        support_labels_onehot = support_labels_onehot.scatter(1, support_labels, 1)
        y_pred = torch.mm(attention, support_labels_onehot)
        clipped_y_pred = y_pred.clamp(EPSILON, 1 - EPSILON)
        return clipped_y_pred


def pairwise_distances(x: torch.Tensor, y: torch.Tensor, matching_fn: str) -> torch.Tensor:
    """
    Efficiently calculate pairwise distances (or other similarity scores) between
    two sets of samples.

    **Arguments**
        x: Query samples. A tensor of shape (n_x, d) where d is the embedding dimension
        y: Class prototypes. A tensor of shape (n_y, d) where d is the embedding dimension
        matching_fn: Distance metric/similarity score to compute between samples
    """
    n_x = x.shape[0]
    n_y = y.shape[0]

    if matching_fn == 'l2':
        distances = (
            x.unsqueeze(1).expand(n_x, n_y, -1) -
            y.unsqueeze(0).expand(n_x, n_y, -1)
        ).pow(2).sum(dim=2)
        return distances
    elif matching_fn == 'cosine':
        normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)
        normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)

        expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)

        cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
        return 1 - cosine_similarities
    elif matching_fn == 'dot':
        expanded_x = x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = y.unsqueeze(0).expand(n_x, n_y, -1)

        return -(expanded_x * expanded_y).sum(dim=2)
    else:
        raise (ValueError('Unsupported similarity function'))
