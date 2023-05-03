from copy import deepcopy
from time import process_time, sleep, time
import traceback
import numpy as np
import torch
from torch import nn
from learn2learn.utils import accuracy
from learn2learn.nn.protonet import compute_prototypes

from fsl.approach.meta_module import LightningMetaModule
from fsl.modules.relation_module import default_relation_module


embedding_dim = {
    'Wang17' : (1,32,32),
    'Lopez17RNN' : (1,10,10),
    'Aceto19MIMETIC' : (1,8,16),
    'Lopez17CNN' : (1,10,20),
    'Lopez17CNNRNN' : (1,10,10),
    'ShallowAutoencoder' : (1,8,10)
}

class LightningRelationNetworks(LightningMetaModule):
    """
    (adapted from: https://github.com/sicara/easy-few-shot-learning)

    **Description**

    A PyTorch Lightning module for Relation Networks.

    **Arguments**

    * **features** (Module) - Feature extractor which classifies input tasks (out shape=(chn,w,h)).
    * **relation_module** (Module) - Module that will output a relation score.
    * **loss** (Function, *optional*, default=MSE) - Loss function which maps the cost of the events.
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

    1. Flood Sung et al. 2018. "Learning to Compare: Relation Network for Few-Shot Learning"
    """

    embedding_shape = (1,24,24)
    mimetic_es = [(1,16,16), (1,16,16), (1,8,16)] # TODO: hard coded
    inner_channels = 8

    def __init__(self, net, relation_module: nn.Module = None, loss=None, **kwargs):
        super(LightningRelationNetworks, self).__init__(**kwargs)
        
        # Meta-Learning specific parameters
        self.loss = loss or nn.MSELoss()
        self.embedding_shape = embedding_dim[kwargs.get("network", None)]
        self.inner_channels = kwargs.get(
            "inner_channels", LightningRelationNetworks.inner_channels)
        assert len(self.embedding_shape) == 3, (
            'Embedding shape must be 3D: (chn, width, height)'
        )
        self.net = net
        self.feature_channel = self.embedding_shape[0]
        self.relation_module = (
            relation_module 
            if relation_module
            else default_relation_module(self.feature_channel, self.inner_channels)
        )

        self.finetune = kwargs['finetune']
        self.no_prototypes = kwargs['no_prototypes']
        self.adaptation_steps = kwargs['adaptation_steps']
        self.finetune_only_comp = kwargs['finetune_only_comp']

        # # For checking purposes
        # self.old_w0 = None
        # self.old_w1 = None

        self.save_hyperparameters({
            "train_ways": self.train_ways,
            "train_shots": self.train_shots,
            "train_queries": self.train_queries,
            "test_ways": self.test_ways,
            "test_shots": self.test_shots,
            "test_queries": self.test_queries,
            "lr": self.lr,
            "lr_strat": self.lr_strat,
            "embedding_shape": self.embedding_shape,
            "inner_channels": self.inner_channels,
            "loss_factor": self.loss_factor,
            "add_cl": self.add_cl,
            'finetune': self.finetune,
            'no_prototypes': self.no_prototypes, 
            'adaptation_steps': self.adaptation_steps,
            'finetune_only_comp': self.finetune_only_comp,
        })

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = LightningMetaModule.add_model_specific_args(parent_parser)
        parser.add_argument(
            "--embedding_shape",
            type=tuple,
            default=LightningRelationNetworks.embedding_shape
        )
        parser.add_argument(
            "--inner_channels",
            type=int,
            default=LightningRelationNetworks.inner_channels
        )
        parser.add_argument(
            '--finetune',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--no_prototypes',
            action='store_true',
            default=False
        )
        parser.add_argument(
            "--adaptation_steps",
            type=int,
            default=10
        )
        parser.add_argument(
            "--finetune_only_comp",
            action='store_true',
            default=False
        )
        return parser

    def meta_learn(self, batch, batch_idx, N, K_s, K_q):
        learner = self.net
        comparator = self.relation_module

        learner.train()
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
        self.embedding_shape = self.mimetic_es[self.stage] if self.net_is_mm else self.embedding_shape

        if not self.training and self.finetune:
            # Apply finetuning
            learner = deepcopy(self.net)
            comparator = deepcopy(self.relation_module)

            full_comp = FullComparator(comparator, K_s, N, self.no_prototypes, self.feature_channel)

            full_comp.train()
            for p in full_comp.parameters():
                p.requires_grad = True
            if self.finetune_only_comp:
                learner.model.eval()
                for p in learner.model.parameters():
                    p.requires_grad = False
            else:
                learner.model.train()
                for p in learner.model.parameters():
                    p.requires_grad = True

            internal_optim = torch.optim.Adam(
                ([] if self.finetune_only_comp else list(learner.model.parameters())) +
                list(full_comp.parameters()), lr=1e-4)

            internal_loss_fn = torch.nn.MSELoss()

            # old_w0 = sum([param.sum().item() for param in learner.model.parameters()])
            # old_w1 = sum([param.sum().item() for param in comparator.parameters()])
            # old_w = sum([param.sum().item() for param in full_comp.parameters()])

            with torch.enable_grad():

                emb_time = time()
                embeddings = learner.model(data[support_indices])
                new_embedding_shape = (embeddings.size(0),) + self.embedding_shape
                # Relation module input should be a feature map(chn, width, height)
                embeddings = embeddings.view(new_embedding_shape)
                embedding_labels = labels[support_indices]
                emb_time = time() - emb_time

                internal_eval_loss = 0

                cmp_time = 0
                for _ in range(self.adaptation_steps):

                    t = time()
                    internal_relation_scores = full_comp(embeddings, embedding_labels)
                    cmp_time += (time() - t)

                    internal_query_labels_onehot = torch.zeros(full_comp.num_samples, N, device=self._device)
                    internal_query_labels_onehot = internal_query_labels_onehot.scatter(
                        1, full_comp.query_labels.unsqueeze(-1), 1)
                    
                    internal_eval_loss += internal_loss_fn(internal_relation_scores, internal_query_labels_onehot)

                bkw_time = time()
                internal_optim.zero_grad()
                internal_eval_loss.backward()
                internal_optim.step()
                bkw_time = time() - bkw_time

            # print()
            # print('*' * 20, 'FINETUNE TIMING', '*' * 20)
            # print(f'{emb_time = }', f'{cmp_time = }', f'{bkw_time = }')
            # print('*' * 57)

            # new_w0 = sum([param.sum().item() for param in learner.model.parameters()])
            # new_w1 = sum([param.sum().item() for param in comparator.parameters()])
            # new_w = sum([param.sum().item() for param in full_comp.parameters()])

            # assert old_w != new_w, f'{old_w} == {new_w}'

            # ASSERTS SUCCESSFULLY PASSED ON [fsl_dev 4b6fca4c]
            # # Check if params update after the finetuning
            # if not self.finetune_only_comp:
            #     assert old_w0 != new_w0, f'{old_w0} == {new_w0}'
            # assert old_w1 != new_w1, f'{old_w1} == {new_w1}'
            # if self.finetune_only_comp:
            #     assert old_w0 == new_w0, f'{old_w0} != {new_w0}'
            # # Check is starting model is the same for each episode during validation/test
            # assert self.old_w0 is None or self.old_w0 == old_w0, f'{self.old_w0} != {old_w0}'
            # assert self.old_w1 is None or self.old_w1 == old_w1, f'{self.old_w1} != {old_w1}'
            # # Check the underlying models used for training
            # if self.finetune_only_comp:
            #     assert sum([param.sum().item() for param in learner.model.parameters()]) == sum([param.sum().item() for param in self.net.model.parameters()])
            # else:
            #     assert sum([param.sum().item() for param in learner.model.parameters()]) != sum([param.sum().item() for param in self.net.model.parameters()])
            # assert sum([param.sum().item() for param in comparator.parameters()]) != sum([param.sum().item() for param in self.relation_module.parameters()])

            # self.old_w0 = old_w0
            # self.old_w1 = old_w1
        # else:
        #     # Reset the checking condition for next validation/test
        #     self.old_w0 = None
        #     self.old_w1 = None

        _, embeddings = learner(
            data, return_features=True, stage=self.stage)
        new_embedding_shape = (embeddings.size(0),) + self.embedding_shape
        # Relation module input should be a feature map(chn, width, height)
        embeddings = embeddings.view(new_embedding_shape)
        support = embeddings[support_indices]
        support_labels = labels[support_indices]
        query = embeddings[query_indices]
        query_labels = labels[query_indices]

        if self.no_prototypes:
            # The approache uses all the support embeddings
            prototypes = support
            # Extending the groud truth
            query_labels = torch.concat([torch.Tensor([i]*K_s*K_q).type_as(query_labels) for i in range(N)])
            num_samples = N*K_s*K_q
        else:
            # RELATION LOGIC
            prototypes = compute_prototypes(support, support_labels)
            num_samples = N*K_q

        # relation_pairs.shape = (n_queries*n_protype, 2*n_channel, width, height)
        relation_pairs = torch.cat(
            (prototypes.unsqueeze(dim=0).expand(query.shape[0], -1, -1, -1, -1),
             query.unsqueeze(dim=1).expand(-1, prototypes.shape[0], -1, -1, -1)
             ), dim=2
        ).view(-1, 2 * self.feature_channel, *query.shape[2:])

        relation_scores = comparator(
            relation_pairs).view(num_samples, N)
        
        # Classification
        query_labels_onehot = torch.zeros(num_samples, N, device=self._device)
        query_labels_onehot = query_labels_onehot.scatter(
            1, query_labels.unsqueeze(-1), 1)
        
        eval_loss = self.loss(relation_scores, query_labels_onehot)
        eval_accuracy = accuracy(relation_scores, query_labels)

        return {
            'loss': eval_loss,
            'accuracy': eval_accuracy,
            'logits': relation_scores,
            'query_labels': query_labels,
            'support_labels': support_labels,
            'le_map': le,
            'support': support.view(-1, support.shape[2]*support.shape[3]),
            'query': query.view(-1, query.shape[2]*query.shape[3])
        }


class FullComparator(nn.Module):
    def __init__(self, comparator, K_s, N, no_prototypes, feature_channel):
        super(FullComparator, self).__init__()
        self.comparator = comparator

        self.K_s = K_s
        self.N = N
        self.no_prototypes = no_prototypes
        self.feature_channel = feature_channel

        self.iK_s = K_s - 1
        self.iK_q = 1

        self.query_labels = None
        self.num_samples = None

    def forward(self, x, labels):

        query_indices = [False] * (self.K_s*self.N)
        for n in range(self.N):
            query_indices[np.random.randint(n * self.K_s, (n+1) * self.K_s)] = True
        query_indices = torch.Tensor(query_indices).type(torch.bool)

        support = x[~query_indices]
        support_labels = labels[~query_indices]
        query = x[query_indices]
        self.query_labels = labels[query_indices]

        if self.no_prototypes:
            # The approache uses all the support embeddings
            prototypes = support
            # Extending the groud truth
            self.num_samples = self.N*self.iK_s*self.iK_q
        else:
            # RELATION LOGIC
            prototypes = compute_prototypes(support, support_labels)
            self.num_samples = self.N*self.iK_q
        
        relation_pairs = torch.cat(
            (prototypes.unsqueeze(dim=0).expand(query.shape[0], -1, -1, -1, -1),
            query.unsqueeze(dim=1).expand(-1, prototypes.shape[0], -1, -1, -1)
            ), dim=2
        ).view(-1, 2 * self.feature_channel, *query.shape[2:])

        relation_scores = self.comparator(relation_pairs).view(self.num_samples, self.N)

        return relation_scores
