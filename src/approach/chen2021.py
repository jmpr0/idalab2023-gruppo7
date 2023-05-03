from argparse import ArgumentParser

import numpy as np
import torch
from datasets.exemplars_dataset import ExemplarsDataset
from torch.utils.data import DataLoader, Dataset

from .incremental_learning import Inc_Learning_Appr


class Appr(Inc_Learning_Appr):
    """Class implementing the Chen2021 proposal in 'Incremental Learning for Mobile Encrypted Traffic Classification'"""

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 logger=None, exemplars_dataset=None, tau=.1, retrain_from_scratch=False, **kwargs):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset, **kwargs)
        self.trn_datasets = []
        self.val_datasets = []
        self.tau = tau
        self.retrain_from_scratch = retrain_from_scratch

        self.model = model
        self.nc_last_task = 0

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--tau', default=.1, type=float,
                            help='Selective-retraining parameter (default=%(default)s)')
        parser.add_argument('--retrain-from-scratch', action='store_true', default=False,

                            help='Retrain from scratch (instead of finetuning) when required (default=%(default)s)')
        return parser.parse_known_args(args)

    def _has_exemplars(self):
        """Returns True in case exemplars are being used"""
        return self.exemplars_dataset is not None and len(self.exemplars_dataset) > 0

    def pre_train_process(self, t, trn_loader):
        """Runs before training all epochs of the task (before the train session)"""
        pass

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        self.is_train = True  # Used to manage different behavior of the function eval

        # Storing trn_loader to enforce selective-retraining
        new_task_trn_loader = trn_loader

        # Add new datasets to existing cumulative ones (line 1 of Algorithm 1)
        if t > 0 and self._has_exemplars():
            self.trn_datasets = [self.exemplars_dataset]
        self.trn_datasets.append(trn_loader.dataset)
        self.val_datasets.append(val_loader.dataset)
        trn_dset = OneVsRestDataset(self.trn_datasets)
        val_dset = OneVsRestDataset(self.val_datasets)
        
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

        # Train only new models (lines 2--5 of Algorithm 1) !! Not efficient: models can be trained in parallel
        for current_target, model in enumerate(self.model.model):
#             if (t > 0 and current_target < self.nc_last_task) or (t == 0 and self.predict_tasks):
            if t > 0 and current_target < self.nc_last_task:
                continue
            print('Train one-vs-rest for the class %s' % current_target)
            self._model = model
            # Setting the target class for the one-vs-rest training
            trn_loader.dataset.set_target(current_target)
            val_loader.dataset.set_target(current_target)
            # continue training as usual
            super().train_loop(t, trn_loader, val_loader)

        # Implement the selective-retraining (lines 6--17 of Algorithm 1)
        if t > 0:
            new_task_trn_loader = DataLoader(new_task_trn_loader.dataset,
                                             batch_size=len(new_task_trn_loader.dataset),
                                             shuffle=False,
                                             num_workers=new_task_trn_loader.num_workers,
                                             pin_memory=new_task_trn_loader.pin_memory)
            n_new_samples = len(new_task_trn_loader.dataset)
            # Computing error ratios for each old model
            err_ratios = []
            for current_target, model in enumerate(self.model.model[:self.nc_last_task]):
                model.eval()
                outputs = None
                for x, _ in new_task_trn_loader:
                    x = self.format_inputs(x)
                    outputs = model(x)[0]
                err_ratios.append((sum(torch.sigmoid(outputs) >= .5).cpu().numpy() / n_new_samples)[0])
            print('Error ratios: %s' % err_ratios)
            # Selective retraining based on error ratio and tau params
            for current_target, err_ratio in enumerate(err_ratios):
                
                # The force_train_from_scratch is needed when the modality is predict_tasks in order to train the first task OvA model
                # This training from scratch procedure is needed only at the first increment (t==1) and only for the base model
                force_train_from_scratch = (self.predict_tasks and t == 1 and current_target == 0)
                
                if (err_ratio >= self.tau) or force_train_from_scratch:
                    if self.retrain_from_scratch or force_train_from_scratch:
                        from networks.network import LLL_Net as Single_Net
                        from copy import deepcopy

                        print('Retraining (from scratch) one-vs-rest for the class %s' % current_target)
                        self.model.model[current_target] = Single_Net(
                            deepcopy(self.model.model), remove_existing_head=True, activate_features=True)
                        self.model.model[current_target].add_head(1)
                        self._model = self.model.model[current_target]
                        self._model.to(self.device)
                    else:
                        print('Retraining (finetuning) one-vs-rest for the class %s' % current_target)
                        self._model = self.model.model[current_target]
                    # Setting the target class for the one-vs-rest training
                    trn_loader.dataset.set_target(current_target)
                    val_loader.dataset.set_target(current_target)
                    # continue training as usual
                    super().train_loop(t, trn_loader, val_loader)

        self.nc_last_task = len(self.model.model)

        self.is_train = False

        return trn_loader

    def post_train_process(self, t, trn_loader, val_loader):
        if t == 0 and len(self.trn_datasets) == 0:
            # Patch to correct the missing behavior when loading pre-trained base model
            self.trn_datasets.append(trn_loader.dataset)
            self.val_datasets.append(val_loader.dataset)

        """Runs after training all the epochs of the task (after the train session)"""
        # EXEMPLAR MANAGEMENT -- select training subset
        trn_loader.dataset.reset_target()
        # The model is set as None because Chen2021 select exemplars from samples, rather than extracted features
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform,
                                                 from_inputs=True)

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self._model.train()
        if self.fix_bn and t > 0:
            self._model.freeze_bn()

        for images, targets in trn_loader:
            images, targets = self.format_inputs(images, targets)
            # Forward current model
            outputs = self._model(images)
            loss = self.criterion(t, outputs, targets)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.clipgrad)
            self.optimizer.step()

    def eval(self, t, val_loader):
        """Override the evaluation code"""
        if self.is_train:  # Preserves the old behavior
            return super().eval(t, val_loader)

        # Computing the returns for each model in the set
        val_dataset = val_loader.dataset
        val_dset = OneVsRestDataset([val_dataset])
        val_loader = DataLoader(val_dset,
                                batch_size=val_loader.batch_size,
                                shuffle=False,
                                num_workers=val_loader.num_workers,
                                pin_memory=val_loader.pin_memory)
        rets = []
        for current_target, model in enumerate(self.model.model):
            # print('Evaluate one-vs-rest for the class %s' % current_target)
            self._model = model
            val_loader.dataset.set_target(current_target)
            rets.append(super().eval(t, val_loader))
        # Join the results from each one-class model
        loss = np.mean([ret[0] for ret in rets])
        outputs_tot = [v.tolist() for v in np.concatenate([ret[3] for ret in rets], axis=1)]  # Not efficient
        # assert np.max(np.sum([ret[4] for target, ret in enumerate(rets)], axis=0)) == 1
        targets_tot = np.sum([[v * target for v in ret[4]] for target, ret in enumerate(rets)],
                             axis=0, dtype=int).tolist()
        acc_taw = np.mean([ret[1] for ret in rets])
        acc_tag = sum([t == p for t, p in zip(targets_tot, np.argmax(outputs_tot, axis=1))]) / len(targets_tot)
        return loss, acc_taw, acc_tag, outputs_tot, targets_tot, None  # TODO: how to store extracted features?

    def calculate_metrics(self, outputs, targets):
        targets = self.format_inputs(targets)
        """Contains the main Task-Aware and Task-Agnostic metrics"""
        pred = torch.zeros_like(targets)
        # Task-Aware Multi-Head & Task-Agnostic Multi-Head (for compatibility)
        for m in range(len(pred)):
            pred[m] = torch.sigmoid(outputs[0][m]).round()
        hits_taw = hits_tag = (pred == targets).float()
        return hits_taw, hits_tag

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        return torch.nn.functional.binary_cross_entropy_with_logits(torch.cat(outputs, dim=1).squeeze(dim=-1), targets)


class OneVsRestDataset(Dataset):
    """Characterizes a dataset for PyTorch -- this dataset returns a one-vs-rest labeling based on target parameter"""

    def __init__(self, datasets):
        self.datasets = datasets
        self._len = sum([len(d) for d in self.datasets])
        self.target = -1
        self.transform = None

    def set_target(self, target):
        self.target = target

    def reset_target(self):
        self.target = -1

    def __len__(self):
        """Denotes the total number of samples"""
        return self._len

    def __getitem__(self, index):
        for d in self.datasets:
            if len(d) <= index:
                index -= len(d)
            else:
                x, y = d[index]
                if self.target != -1:
                    return x, 1. if y == self.target else 0.
                else:
                    return x, y
