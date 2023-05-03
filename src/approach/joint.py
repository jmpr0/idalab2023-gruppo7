from argparse import ArgumentParser
from copy import deepcopy

import torch
from datasets.exemplars_dataset import ExemplarsDataset
from torch.utils.data import DataLoader, Dataset

from .incremental_learning import Inc_Learning_Appr


class Appr(Inc_Learning_Appr):
    """Class implementing the joint baseline"""

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 logger=None, exemplars_dataset=None, freeze_after=-1, balance_new_tasks=False, **kwargs):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset, **kwargs)
        self.trn_datasets = []
        self.val_datasets = []
        self.freeze_after = freeze_after
        self.balance_new_tasks = balance_new_tasks

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--freeze-after', default=-1, type=int, required=False,
                            help='Freeze model except heads after the specified task'
                                 '(-1: normal Incremental Joint Training, no freeze) (default=%(default)s)')
        parser.add_argument('--balance-new-tasks', action='store_true',
                           help='Balancing of new classes samples to match the number'
                            'of samples in memory (default=%(default)s)')
        return parser.parse_known_args(args)

    def _has_exemplars(self):
        """Returns True in case exemplars are being used"""
        return self.exemplars_dataset is not None and len(self.exemplars_dataset) > 0

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        # add new datasets to existing cumulative ones
        if t > 0 and self._has_exemplars():
            # if has exemplars, the old classes training dataset is taken from the memory
            self.trn_datasets = [self.exemplars_dataset]
        
        if t == 0 or not self.balance_new_tasks:
            # if it is not balanced, or it is the first training task, simply append the entire dataset
            self.trn_datasets.append(trn_loader.dataset)
        else:
            # balancing new classes basing on how much samples per class are stored into the memory
            num_exemplars_per_class = self.exemplars_dataset.max_num_exemplars_per_class
            if not num_exemplars_per_class:
                import math
                num_exemplars_per_class = math.ceil(
                    self.exemplars_dataset.max_num_exemplars / sum(self.model.task_cls[:t]))

            new_task_exemplars_dataset = ExemplarsDataset(
                trn_loader.dataset.transform, trn_loader.dataset.class_indices[self.model.task_cls[t - 1]:],
                num_exemplars_per_class=num_exemplars_per_class, exemplar_selection='random',
                is_networking=self.exemplars_dataset.is_networking, base_class_index=sum(self.model.task_cls[:t])
            )
            
            # retrieve random balancing selection of new classes
            new_task_exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)
            self.trn_datasets.append(new_task_exemplars_dataset)
        
        self.val_datasets.append(val_loader.dataset)
        trn_dset = JointDataset(self.trn_datasets)
        val_dset = JointDataset(self.val_datasets)
        trn_loader = DataLoader(trn_dset,
                                batch_size=trn_loader.batch_size,
                                shuffle=True,
                                num_workers=trn_loader.num_workers,
                                pin_memory=trn_loader.pin_memory)
        val_loader = DataLoader(val_dset,
                                batch_size=val_loader.batch_size,
                                shuffle=False,
                                num_workers=val_loader.num_workers,
                                pin_memory=val_loader.pin_memory)
        # continue training as usual
        return super().train_loop(t, trn_loader, val_loader)

    def post_train_process(self, t, trn_loader, val_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        if t == 0 and len(self.trn_datasets) == 0:
            # Patch to correct the missing behavior when loading pre-trained base model
            self.trn_datasets.append(trn_loader.dataset)
            self.val_datasets.append(val_loader.dataset)

        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

        if -1 < self.freeze_after <= t:
            self.model.freeze_all()
            for head in self.model.heads:
                for param in head.parameters():
                    param.requires_grad = True

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        if self.freeze_after < 0 or t <= self.freeze_after:
            self.model.train()
            if self.fix_bn and t > 0:
                self.model.freeze_bn()
        else:
            self.model.eval()
            for head in self.model.heads:
                head.train()
        for images, targets in trn_loader:
            images, targets = self.format_inputs(images, targets)
            # Forward current model
            outputs = self.model(images)
            loss = self.criterion(t, outputs, targets)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        return torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)


class JointDataset(Dataset):
    """Characterizes a dataset for PyTorch -- this dataset accumulates each task dataset incrementally"""

    def __init__(self, datasets):
        self.datasets = datasets
        self._len = sum([len(d) for d in self.datasets])
        self.transform = None

    def __len__(self):
        'Denotes the total number of samples'
        return self._len

    def __getitem__(self, index):
        for d in self.datasets:
            if len(d) <= index:
                index -= len(d)
            else:
                x, y = d[index]
                return x, y
