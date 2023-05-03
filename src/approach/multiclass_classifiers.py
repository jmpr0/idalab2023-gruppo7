from argparse import ArgumentParser

import numpy as np
from sklearn.compose import TransformedTargetRegressor
import torch
from datasets.exemplars_dataset import ExemplarsDataset
from torch.utils.data import DataLoader, Dataset

from .incremental_learning import Inc_Learning_Appr, tensor_to_cpu


class Appr(Inc_Learning_Appr):
    """Class implementing the Multiclass Classifiers approach'"""

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 logger=None, exemplars_dataset=None, tau=.1, **kwargs):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset, **kwargs)
        self.trn_datasets = []
        self.val_datasets = []
        self.tau = tau
        
        self.model = model
        self.nc_last_task = 0
        self.offset = 0

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--tau', default=.1, type=float,
                            help='Selective-retraining parameter (default=%(default)s)')
        return parser.parse_known_args(args)

    def _has_exemplars(self):
        """Returns True in case exemplars are being used"""
        return self.exemplars_dataset is not None and len(self.exemplars_dataset) > 0
    
    def train(self, t, trn_loader, val_loader, base_model_path=None):
        if not self.is_ml:
            super().train(t, trn_loader, val_loader, base_model_path)
        else:
            self.model.fit(t, trn_loader.dataset.images, trn_loader.dataset.labels)

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
        #self.trn_datasets.append(trn_loader.dataset)
        #self.val_datasets.append(val_loader.dataset)
        self.trn_datasets = [trn_loader.dataset]
        self.val_datasets = [val_loader.dataset]

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
        
        max_target = np.max(trn_loader.dataset.datasets[0].labels) - trn_loader.dataset.offset        
#         print(max_target)
        if max_target != 0:
        # Train only new models (lines 2--5 of Algorithm 1) !! Not efficient: models can be trained in parallel
            for current_target, model in enumerate(self.model.model):
                if t > 0 and current_target < self.nc_last_task:
                    continue
                #print('Train one-vs-rest for the class %s' % current_target)
                print('Train model - Episode %s' % current_target)

                self._model = model  # self._model is used to store the current model from the ModuleList self.model.model
                # Setting the target class for the one-vs-rest training
                #trn_loader.dataset.set_target(current_target)
                #val_loader.dataset.set_target(current_target)
                trn_loader.dataset.reset_target()
                val_loader.dataset.reset_target()

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
        
    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn(t)
        
        max_target = np.max(trn_loader.dataset.datasets[0].labels) - trn_loader.dataset.offset        

        for images, targets in trn_loader:
            if max_target != 0:
                images, targets = self.format_inputs(images, targets)
                # Forward current model
                outputs = self._model(images)
                loss = self.criterion(t, outputs, targets)
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
                self.optimizer.step()

    def eval(self, t, val_loader):
        if not self.is_ml:
            
            """Override the evaluation code"""
            if self.is_train:  # Preserves the old behavior
                return super().eval(t, val_loader)

            # Computing the returns for each model in the set
            val_dataset = val_loader.dataset
            val_dset = OneVsRestDataset([val_dataset])
            val_dset.set_target(0)
            val_loader = DataLoader(val_dset,
                                    batch_size=val_loader.batch_size,
                                    shuffle=False,
                                    num_workers=val_loader.num_workers,
                                    pin_memory=val_loader.pin_memory)
            rets = []

            val_loader.dataset.set_target(0)
            """Contains the evaluation code"""        
            outputs_tot, targets_tot, features_tot = [], [], []
            with torch.no_grad():
                total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
                self.model.eval()
                for images, targets in val_loader:
                    images, targets = self.format_inputs(images, targets)
                    # Forward current model
                    outputs = self.model(images, return_features=False)
                    outputs = [output[0] for output in outputs]
                    outputs = [torch.cat(outputs, dim=1)]
                    features_tot.extend([])
                    outputs_tot.extend(np.concatenate(tensor_to_cpu(outputs), axis=-1).tolist())
                    targets_tot.extend(targets.tolist())
                    loss = self.criterion(t, outputs, targets)
                    hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                    # Log
                    total_loss += loss.item() * len(targets)
                    total_acc_taw += hits_taw.sum().item()
                    total_acc_tag += hits_tag.sum().item()
                    total_num += len(targets)
                rets.append([total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num, outputs_tot, targets_tot, features_tot])
            # Join the results from each one-class model
            loss = np.mean([ret[0] for ret in rets])
            outputs_tot = [v.tolist() for v in np.concatenate([ret[3] for ret in rets], axis=1)]  # Not efficient
            print(np.shape(outputs_tot))
            # assert np.max(np.sum([ret[4] for target, ret in enumerate(rets)], axis=0)) == 1
            targets_tot = np.sum([[v for v in ret[4]] for target, ret in enumerate(rets)],axis=0, dtype=int).tolist()
            #targets_tot = np.sum([[v * target for v in ret[4]] for target, ret in enumerate(rets)],axis=0, dtype=int).tolist()
            acc_taw = np.mean([ret[1] for ret in rets])
            acc_tag = sum([t == p for t, p in zip(targets_tot, np.argmax(outputs_tot, axis=1))]) / len(targets_tot)
            extracted_features = None
            
            return loss, acc_taw, acc_tag, outputs_tot, targets_tot, extracted_features  # TODO: how to store extracted features?
        else:
            retss = self.model.predict(t, val_loader.dataset.images)
            true = val_loader.dataset.labels
            rets = np.concatenate(retss, axis=1)
            # TODO: compute the rest as usual for the tag
            # TODO: select the proper model from retss for taw
            assert len(rets) == len(true)
            preds_tag = np.argmax(rets, axis=1)
            predss = np.array([np.argmax(r, axis=1) for r in retss])
            preds_taw = np.array([predss[0]] + [
                ps + np.ones(ps.shape) * np.sum([len(retss[j][0]) for j in range(i + 1)], )
                for i, ps in enumerate(predss[1:])])[t]
            loss = np.nan
            targets_tot = true
            acc_tag = sum([t == p for t, p in zip(true, preds_tag)]) / len(true)
            acc_taw = sum([t == p for t, p in zip(true, preds_taw)]) / len(true)
            outputs_tot = [[r.tolist() for r in rs] for rs in np.concatenate(retss, axis=1)]
            extracted_features = None
            from sklearn.metrics import f1_score
            pc_f1_tag = f1_score(true, preds_tag, average=None)
            true_idx = [i for i, t in enumerate(sorted(set(np.concatenate((true, preds_tag))))) if t in set(true)]
            pc_f1_tag = [pc_f1_tag[i] for i in true_idx]
            print('f1_tag:', np.mean(pc_f1_tag))
            print('f1_taw:', f1_score(true, preds_taw, average='macro'))
            print('per-class f1_tag:', pc_f1_tag)
            print('per-class f1_taw:', f1_score(true, preds_taw, average=None))
            return loss, acc_taw, acc_tag, outputs_tot, targets_tot, extracted_features

    def calculate_metrics(self, outputs, targets):
        targets = self.format_inputs(targets)
        """Contains the main Task-Aware and Task-Agnostic metrics"""
        pred = torch.zeros_like(targets)
        
        # Task-Aware Multi-Head & Task-Agnostic Multi-Head (for compatibility)
        '''
        for m in range(len(pred)):
            print(outputs[0][m])
            print(outputs[0][m].size(dim=0))
            pred[m] = torch.special.expit(outputs[0][m]).round()
            print(pred[m])
            input()
        '''
        temp = torch.sigmoid(outputs[0])
        pred = torch.argmax(temp,dim=1)
        hits_taw = hits_tag = (pred == targets).float()
        return hits_taw, hits_tag

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        if len(outputs[0][0]) == 1:
            return torch.nn.functional.binary_cross_entropy_with_logits(torch.cat(outputs, dim=1).squeeze(dim=-1), targets)  
        else:
            return torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1).squeeze(dim=-1), targets)
        

class OneVsRestDataset(Dataset):
    """Characterizes a dataset for PyTorch -- this dataset returns a one-vs-rest labeling based on target parameter"""

    def __init__(self, datasets):
        self.datasets = datasets
        self._len = sum([len(d) for d in self.datasets])
        self.target = None
        self.transform = None
        self.offset = np.min(self.datasets[0].labels)

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
                    return x, y
                else:
                    return x, y - self.offset
