import warnings
from argparse import ArgumentParser
from copy import deepcopy
from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union

import numpy as np
import torch
from datasets.exemplars_dataset import ExemplarsDataset
from datasets.exemplars_selection import override_dataset_transform
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

from .incremental_learning import Inc_Learning_Appr, tensor_to_cpu


class ExperienceReplaySampler(Sampler):
    """
    Class implementing the Experience Replay (ER) mini batching proposed in https://tajanthan.github.io/il/docs/cler.pdf
    """

    def __init__(self, batch_size, mem_samples_index, new_samples_index, generator=None) -> None:
        # mem~new proportion of 2~8
        self.new_batch_size = int(batch_size * .8)
        self.mem_batch_size = batch_size - self.new_batch_size
        self.mem_samples_index = mem_samples_index
        self.new_samples_index = new_samples_index
        self.generator = generator

        # Setting the number of batches as the minimum number of mini batches for memory and new samples
        self.num_batches = min([
            len(self.mem_samples_index) // self.mem_batch_size,
            len(self.new_samples_index) // self.new_batch_size])

    def __iter__(self) -> Iterator[int]:
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        # Shuffling the memory and new samples indexes to obtain shuffle=True
        self.mem_samples_index = self.mem_samples_index[
            torch.randperm(len(self.mem_samples_index), generator=generator)]
        self.new_samples_index = self.new_samples_index[
            torch.randperm(len(self.new_samples_index), generator=generator)]

        # Act as drop_last=True
        for i in range(self.num_batches):
            yield from self.mem_samples_index[i * self.mem_batch_size:(i + 1) * self.mem_batch_size]
            yield from self.new_samples_index[i * self.new_batch_size:(i + 1) * self.new_batch_size]

    def __len__(self) -> int:
        return len(self.mem_samples_index) + len(self.new_samples_index)


class Appr(Inc_Learning_Appr):
    """Class implementing the SS-IL (Separated Softmax-IL) approach described in https://arxiv.org/pdf/2003.13947.pdf
    Original code available at https://github.com/hongjoon0805/SS-IL-Official
    """

    def __init__(self, model, device, nepochs=60, lr=0.5, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0.9, wd=1e-5, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, logger=None, exemplars_dataset=None, lamb=1, T=2, knowledge_distillation=None,
                 **kwargs):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset, **kwargs)
        self.model_old = None
        self.lamb = lamb
        self.T = T
        self.knowledge_distillation = getattr(self, knowledge_distillation)
        self.mem_batch_size = 0

        # iCaRL is expected to be used with exemplars. If needed to be used without exemplars, overwrite here the
        # `_get_optimizer` function with the one in LwF and update the criterion
        have_exemplars = self.exemplars_dataset.max_num_exemplars + self.exemplars_dataset.max_num_exemplars_per_class
        if not have_exemplars:
            warnings.warn("Warning: iCaRL is expected to use exemplars. Check documentation.")

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
        parser.add_argument('--T', default=2, type=int, required=False,
                            help='Temperature scaling (default=%(default)s)')
        parser.add_argument('--knowledge-distillation', default='tkd', type=str, choices=['gkd', 'tkd'],
                            help='Knowledge distillation used (default=%(default)s)')
        return parser.parse_known_args(args)

    # Algorithm 1: iCaRL NCM Classify
    def classify(self, task, features, targets):
        # expand means to all batch images
        means = torch.stack(self.exemplar_means)
        means = torch.stack([means] * features.shape[0])
        means = means.transpose(1, 2)
        # expand all features to all classes
        features = features / features.norm(dim=1).view(-1, 1)
        features = features.unsqueeze(2)
        features = features.expand_as(means)
        # get distances for all images to all exemplar class means -- nearest prototype
        dists = (features - means).pow(2).sum(1)
        if len(dists.size()) > 2:
            dists = dists.squeeze()
        # Task-Aware Multi-Head
        num_cls = self.model.task_cls[task]
        offset = self.model.task_offset[task]
        pred = dists[:, offset:offset + num_cls].argmin(1)
        hits_taw = (pred + offset == targets).float()
        # Task-Agnostic Multi-Head
        pred = dists.argmin(1)
        hits_tag = (pred == targets).float()
        return hits_taw, hits_tag, dists

    def compute_mean_of_exemplars(self, trn_loader, transform):
        # change transforms to evaluation for this calculation
        with override_dataset_transform(self.exemplars_dataset, transform) as _ds:
            # change dataloader so it can be fixed to go sequentially (shuffle=False), this allows to keep same order
            icarl_loader = DataLoader(_ds, batch_size=trn_loader.batch_size, shuffle=False,
                                      num_workers=trn_loader.num_workers, pin_memory=trn_loader.pin_memory)
            # extract features from the model for all train samples
            # Page 2: "All feature vectors are L2-normalized, and the results of any operation on feature vectors,
            # e.g. averages are also re-normalized, which we do not write explicitly to avoid a cluttered notation."
            extracted_features = []
            extracted_targets = []
            with torch.no_grad():
                self.model.eval()
                for images, targets in icarl_loader:
                    images = self.format_inputs(images)
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
            er_sampler = ExperienceReplaySampler(
                batch_size=trn_loader.batch_size,
                new_samples_index=np.arange(len(trn_loader.dataset)),
                mem_samples_index=np.arange(
                    len(trn_loader.dataset), len(trn_loader.dataset) + len(self.exemplars_dataset)),
            )
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory,
                                                     sampler=er_sampler)
            self.mem_batch_size = er_sampler.mem_batch_size

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
                # Forward old model
                targets_old = None
                if t > 0:
                    targets_old = self.model_old(images)
                # Forward current model
                outputs, features = self.model(images, return_features=True)
                features_tot.extend(features.tolist())
                outputs_tot.extend(np.concatenate(tensor_to_cpu(outputs), axis=1).tolist())
                targets_tot.extend(targets.tolist())
                loss = self.criterion(t, outputs, targets, targets_old, is_training=False)
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # Log
                total_loss += loss.data.cpu().numpy().item() * len(targets)
                total_acc_taw += hits_taw.sum().data.cpu().numpy().item()
                total_acc_tag += hits_tag.sum().data.cpu().numpy().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num, outputs_tot, targets_tot, features_tot

    def cross_entropy(self, outputs, targets, exp=1.0, size_average=True, eps=1e-5):
        """Calculates cross-entropy with temperature scaling"""
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
    def criterion(self, t, outputs, targets, outputs_old=None, is_training=True):
        """Returns the loss value"""
        # try:
        #     print('New', min(targets[self.mem_batch_size:]), max(targets[self.mem_batch_size:]))
        # except:
        #     pass
        # try:
        #     print('Mem', min(targets[:self.mem_batch_size]), max(targets[:self.mem_batch_size]))
        # except:
        #     pass

        if is_training:
            # Classification loss for new classes outputs
            # Outputs[i] contains mem_batch_size elements from the memory and others from new classes
            # when t==0, self.mem_batch_size is zero, in order to consider all the classes as new (base) classes
            loss = torch.nn.functional.cross_entropy(
                outputs[t][self.mem_batch_size:], (targets - self.model.task_offset[t])[self.mem_batch_size:])
        else:
            # Classification loss for the entire classes space
            loss = torch.nn.functional.cross_entropy(torch.cat(outputs[:t + 1], axis=1), targets)

        # Distillation loss for old classes
        if t > 0:
            if is_training:
                loss += torch.nn.functional.cross_entropy(
                    torch.cat(outputs[:t], axis=1)[:self.mem_batch_size], targets[:self.mem_batch_size])

            loss += self.lamb * self.knowledge_distillation(outputs_old, outputs, t)
        return loss
