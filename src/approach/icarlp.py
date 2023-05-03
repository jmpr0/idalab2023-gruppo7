import warnings
from argparse import ArgumentParser
from copy import deepcopy

import numpy as np
import torch
from datasets.exemplars_dataset import ExemplarsDataset
from datasets.exemplars_selection import override_dataset_transform
from torch.utils.data import DataLoader

from .incremental_learning import Inc_Learning_Appr, tensor_to_cpu


class Appr(Inc_Learning_Appr):
    """Class implementing the Incremental Classifier and Representation Learning (iCaRL+) approach
    described in https://arxiv.org/abs/1611.07725
    Original code available at https://github.com/srebuffi/iCaRL
    """

    def __init__(self, model, device, nepochs=60, lr=0.5, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0.9, wd=1e-5, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, logger=None, exemplars_dataset=None, lamb=1, knowledge_distillation=None,
                 **kwargs):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset, **kwargs)
        self.model_old = None
        self.lamb = lamb
        self.knowledge_distillation = getattr(self, knowledge_distillation)

        # iCaRL is expected to be used with exemplars. If needed to be used without exemplars, overwrite here the
        # `_get_optimizer` function with the one in LwF and update the criterion
        have_exemplars = self.exemplars_dataset.max_num_exemplars + self.exemplars_dataset.max_num_exemplars_per_class
        if not have_exemplars:
            warnings.warn("Warning: iCaRL+ is expected to use exemplars. Check documentation.")

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # Sec. 4. " allowing iCaRL to balance between CE and distillation loss."
        parser.add_argument('--lamb', default=1, type=float, required=False,
                            help='Forgetting-intransigence trade-off (default=%(default)s)')
        parser.add_argument('--knowledge-distillation', default='gkd', type=str, choices=['gkd', 'tkd'],
                            help='Knowledge distillation used (default=%(default)s)')
        return parser.parse_known_args(args)

    # # Algorithm 1: iCaRL NCM Classify
    # def classify(self, task, features, targets):
    #     # expand means to all batch images
    #     means = torch.stack(self.exemplar_means)
    #     means = torch.stack([means] * features.shape[0])
    #     means = means.transpose(1, 2)
    #     # expand all features to all classes
    #     features = features / features.norm(dim=1).view(-1, 1)
    #     features = features.unsqueeze(2)
    #     features = features.expand_as(means)
    #     # get distances for all images to all exemplar class means -- nearest prototype
    #     dists = (features - means).pow(2).sum(1).squeeze()
    #     # Task-Aware Multi-Head
    #     num_cls = self.model.task_cls[task]
    #     offset = self.model.task_offset[task]
    #     pred = dists[:, offset:offset + num_cls].argmin(1)
    #     hits_taw = (pred + offset == targets.to(self.device)).float()
    #     # Task-Agnostic Multi-Head
    #     pred = dists.argmin(1)
    #     hits_tag = (pred == targets.to(self.device)).float()
    #     return hits_taw, hits_tag

    def compute_mean_of_exemplars(self, trn_loader, transform):
        # change transforms to evaluation for this calculation
        with override_dataset_transform(self.exemplars_dataset, transform) as _ds:
            # change dataloader so it can be fixed to go sequentially (shuffle=False), this allows to keep same order
            icarlp_loader = DataLoader(_ds, batch_size=trn_loader.batch_size, shuffle=False,
                                       num_workers=trn_loader.num_workers, pin_memory=trn_loader.pin_memory)
            # extract features from the model for all train samples
            # Page 2: "All feature vectors are L2-normalized, and the results of any operation on feature vectors,
            # e.g. averages are also re-normalized, which we do not write explicitly to avoid a cluttered notation."
            extracted_features = []
            extracted_targets = []
            with torch.no_grad():
                self.model.eval()
                for images, targets in icarlp_loader:
                    images, targets = self.format_inputs(images, targets)
                    feats = self.model(images, return_features=True)[1]
                    # normalize
                    extracted_features.append(feats / feats.norm(dim=1).view(-1, 1))
                    extracted_targets.extend(targets)
            extracted_features = torch.cat(extracted_features)
            extracted_targets = np.array(extracted_targets)
            for curr_cls in np.unique(extracted_targets):
                # get all indices from current class
                cls_ind = np.where(extracted_targets == curr_cls)[0]
                # get all extracted features for current class
                cls_feats = extracted_features[cls_ind]
                # add the exemplars to the set and normalize
                cls_feats_mean = cls_feats.mean(0) / cls_feats.mean(0).norm()
                self.exemplar_means.append(cls_feats_mean)

    def pre_train_process(self, t, trn_loader):
        """Runs before training all epochs of the task (before the train session)"""
        # remove mean of exemplars during training since Alg. 1 is not used during Alg. 2
        self.exemplar_means = []

    # Algorithm 2: iCaRL Incremental Train
    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        # Algorithm 3: iCaRL Update Representation
        # Alg. 3. "form combined training set", add exemplars to train_loader
        if t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)

        # FINETUNING TRAINING -- contains the epochs loop
        return super().train_loop(t, trn_loader, val_loader)

    def post_train_process(self, t, trn_loader, val_loader):
        """Runs after training all the epochs of the task (after the train session)"""

        # EXEMPLAR MANAGEMENT -- select training subset
        # Algorithm 4: iCaRL ConstructExemplarSet and Algorithm 5: iCaRL ReduceExemplarSet
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

        # TODO: check (and remove eventually) the need for exemplars' mean computation
        # compute mean of exemplars
        self.compute_mean_of_exemplars(trn_loader, val_loader.dataset.transform)

        # Save old model to extract features later. This is different from the original approach, since they propose to
        #  extract the features and store them for future usage. However, when using data augmentation, it is easier to
        #  keep the model frozen and extract the features when needed.
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_loader:
            images, targets = self.format_inputs(images, targets)
            # Forward old model
            outputs_old = None
            if t > 0:
                outputs_old = self.model_old(images)
            # Forward current model
            outputs = self.model(images)
            loss = self.criterion(t, outputs, targets, outputs_old)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        outputs_tot, targets_tot, features_tot = [], [], []
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                images, targets = self.format_inputs(images, targets)
                """
                From LWF (lwf.py)
                """
                # Forward old model
                targets_old = None
                if t > 0:
                    targets_old = self.model_old(images)
                # Forward current model
                outputs, features = self.model(images, return_features=True)
                features_tot.extend(features.tolist())
                outputs_tot.extend(np.concatenate(tensor_to_cpu(outputs), axis=1).tolist())
                targets_tot.extend(targets.tolist())
                loss = self.criterion(t, outputs, targets, targets_old)
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # Log
                total_loss += loss.data.cpu().numpy().item() * len(targets)
                total_acc_taw += hits_taw.sum().data.cpu().numpy().item()
                total_acc_tag += hits_tag.sum().data.cpu().numpy().item()
                total_num += len(targets)
                # # Forward old model
                # outputs_old = None
                # if t > 0:
                #     outputs_old = self.model_old(images.to(self.device))
                # # Forward current model
                # outputs, feats = self.model(images.to(self.device), return_features=True)
                # loss = self.criterion(t, outputs, targets.to(self.device), outputs_old)
                # # during training, the usual accuracy is computed on the outputs
                # if not self.exemplar_means:
                #     hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # else:
                #     hits_taw, hits_tag = self.classify(t, feats, targets)
                # # Log
                # total_loss += loss.item() * len(targets)
                # total_acc_taw += hits_taw.sum().item()
                # total_acc_tag += hits_tag.sum().item()
                # total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num, outputs_tot, targets_tot, features_tot

    def cross_entropy(self, outputs, targets, exp=1.0, size_average=True, eps=1e-5):
        """Calculates cross-entropy with temperature scaling"""
        """
        From LWF (lwf.py)
        """
        out = torch.nn.functional.softmax(outputs, dim=1)
        tar = torch.nn.functional.softmax(targets, dim=1)
        if exp != 1:
            out = out.pow(exp)
            out = out / out.sum(1).view(-1, 1).expand_as(out)
            tar = tar.pow(exp)
            tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
        out = out + eps / out.size(1)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        ce = -(tar * out.log()).sum(1)
        if size_average:
            ce = ce.mean()
        return ce

    def gkd(self, outputs_old, outputs, t):
        return self.cross_entropy(torch.cat(outputs[:t], dim=1), torch.cat(outputs_old[:t], dim=1), exp=1.0 / self.T)

    def tkd(self, outputs_old, outputs, t):
        return sum(self.cross_entropy(outputs[j], outputs_old[j], exp=1.0 / self.T) for j in range(t))

    # Algorithm 3: classification and distillation terms -- original formulation has no trade-off parameter (lamb=1)
    def criterion(self, t, outputs, targets, outputs_old=None):
        """Returns the loss value"""

        # Classification loss for new classes
        loss = torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        # Distillation loss for old classes
        if t > 0:
            # Forcing temperature to 1
            self.T = 1
            """
            From LWF (lwf.py)
            """
            # Knowledge distillation loss for all previous tasks
            loss += self.lamb * self.knowledge_distillation(outputs_old, outputs, t)
        return loss
