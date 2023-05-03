from argparse import ArgumentParser

import torch
import time
import numpy as np
from copy import deepcopy
from datasets.exemplars_dataset import ExemplarsDataset

from .incremental_learning import Inc_Learning_Appr
from util.transformers import duplicating, jittering, traslating, slicing, features_hiding


class Appr(Inc_Learning_Appr):
    """Class implementing the finetuning baseline"""

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 logger=None, exemplars_dataset=None, num_ft_epochs=100, T=1, lamb=1, ft_lr=0.1, ft_lr_min=1e-4, ft_lr_patience=5,
                 ft_lr_factor=3, knowledge_distillation=None, augment=False, augment_type='', augment_after=2, augment_factor=1,
                 no_halluc=False, halluc_after=1, freeze_after=0, **kwargs):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset, **kwargs)

        self.ft_epochs = num_ft_epochs
        self.T = T
        self.lamb = lamb
        self.ft_lr = ft_lr
        self.ft_lr_min = ft_lr_min
        self.ft_lr_factor = ft_lr_factor
        self.ft_lr_patience = ft_lr_patience
        self.knowledge_distillation = getattr(self, knowledge_distillation)
        self.augment = augment
        self.augment_type = 'a' if 'a' in augment_type else augment_type
        self.augment_after = augment_after
        self.augment_factor = augment_factor
        self.no_halluc = no_halluc
        self.halluc_after = halluc_after
        self.freeze_after = freeze_after

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--lamb', default=1, type=float, required=False,
                            help='Forgetting-intransigence trade-off (default=%(default)s)')
        parser.add_argument('--T', default=1, type=int, required=False,
                            help='Temperature scaling (default=%(default)s)')
        parser.add_argument('--num-ft-epochs', default=5, type=int, required=False,
                            help='Number of epochs for training bias (default=%(default)s)')
        parser.add_argument('--ft-lr', default=0.01, type=float, required=False,
                            help='Finetuning LR (default=%(default)s)')
        parser.add_argument('--ft-lr-min', default=1e-4, type=float, required=False,
                            help='Finetuning minimum LR for early stopping (default=%(default)s)')
        parser.add_argument('--ft-lr-factor', default=3, type=float, required=False,
                            help='Finetuning LR decay factor (default=%(default)s)')
        parser.add_argument('--ft-lr-patience', default=5, type=int, required=False,
                            help='Finetuning LR patience (default=%(default)s)')
        parser.add_argument('--knowledge-distillation', default='gkd', type=str, choices=['gkd', 'tkd'],
                            help='Knowledge distillation used (default=%(default)s)')
        parser.add_argument('--augment', action='store_true',
                            help='Apply new samples\' augmentation based on transforms (default=%(default)s)')
        parser.add_argument('--augment-type', default='', type=str,
                            help='Type of augmentation: (a)ll, (d)uplicate, (j)itter, (t)raslate, (s)lice, and features_(h)ide (default=%(default)s)')
        parser.add_argument('--augment-factor', default=1, type=int, required=False,
                            help='Number of times augmetation is applied (default=%(default)s)')
        parser.add_argument('--augment-after', default=1, type=int, required=False,
                            help='Episode from which the new training set is augmented (default=%(default)s)')
        parser.add_argument('--no-halluc', action='store_true',
                            help='Do not apply hallucination to memory samples (default=%(default)s)')
        parser.add_argument('--halluc-after', default=1, type=int, required=False,
                            help='Episode from which the hallucinated training is performed (default=%(default)s)')
        parser.add_argument('--freeze-after', default=0, type=int, required=False,
                            help='Episode from which the backbone is freezed (default=%(default)s)')
        return parser.parse_known_args(args)

    def _get_optimizer(self, t=None):
        """Returns the optimizer"""
        params = self.model.parameters()
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        # print(trn_loader.__dict__)
        # input()

        new_trn_loader = trn_loader

        # add exemplars to train_loader
        if t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)

        # If --augment, the new_trn_loader is augmented via transformations
        if self.augment and t >= self.augment_after:
            images = new_trn_loader.dataset.images
            labels = new_trn_loader.dataset.labels
            seeds = new_trn_loader.dataset.seeds
            new_trn_loader = deepcopy(new_trn_loader)

            multiplier = 1
            augmented_images = []
            for atype in self.augment_type:
                if atype in ['a', 'd']:
                    # Duplicating the samples augment_factor times
                    augmented_images += [
                        duplicating(x) for _ in range(self.augment_factor) for x in images]
                    multiplier += self.augment_factor
                if atype in ['a', 'j']:
                    # Jittering samples' IAT feature augment_factor times
                    augmented_images += [
                        jittering(x, features=[1], random_state=s) for _ in range(self.augment_factor) for x, s in zip(images, seeds)]
                    multiplier += self.augment_factor
                if atype in ['a', 't']:
                    # Traslating samples' IAT feature augment_factor times
                    augmented_images += [
                        traslating(x, features=[1], random_state=s) for _ in range(self.augment_factor) for x, s in zip(images, seeds)]
                    multiplier += self.augment_factor
                if atype in ['a', 's']:
                    # Slicing the first (.5) and the second (0) half of each sample
                    augmented_images += [
                        slicing(x, wr=.5, pr=pr) for pr in [.5, 0] for x in images]
                    multiplier += 2
                if atype in ['a', 'h']:
                    # Hiding each feature for each sample
                    augmented_images += [
                        features_hiding(x, features=[feature]) for feature in range(4) for x in images]
                    multiplier += 4

            new_trn_loader.dataset.images = (images + augmented_images)
            new_trn_loader.dataset.labels = labels * multiplier

        # Pretraining (t==0) and Episode training phase 1 (t>0)
        super().train_loop(t, new_trn_loader, val_loader)

        if not self.no_halluc and t >= self.halluc_after:
            # Episode training phase 2

            # Memory DataLoader w/ len(dataset) batch size
            mem_loader = torch.utils.data.DataLoader(self.exemplars_dataset,
                                                     batch_size=len(self.exemplars_dataset),
                                                     shuffle=False,
                                                     num_workers=new_trn_loader.num_workers,
                                                     pin_memory=new_trn_loader.pin_memory)

            # Storing old_model for pseudo_labels of the samples stored in memory
            old_model = deepcopy(self.model)
            old_model.freeze_all()

            # Episode training phase 3

            # Model unfreezing
            self.model.unfreeze_all()
            # Old heads freezing
            self.model.freeze_heads(range(t))

            # Extended DataLoader containing both memory (old) and new classes' exemplars
            ext_loader = torch.utils.data.DataLoader(new_trn_loader.dataset + mem_loader.dataset,
                                                     batch_size=new_trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=new_trn_loader.num_workers,
                                                     pin_memory=new_trn_loader.pin_memory)

            """Contains the epochs loop"""
            lr = self.ft_lr
            best_loss = np.inf
            patience = self.ft_lr_patience

            self.optimizer = self._get_optimizer(t)
            self.optimizer.param_groups[0]['lr'] = lr

            best_model = self.model.get_copy()

            # Backbone and new head finetuning
            self.model.train()
            # Loop epochs
            for e in range(self.ft_epochs):
                # Train
                tot_loss = []
                clock0 = time.process_time()
                for images, targets in ext_loader:
                    images, targets = self.format_inputs(images, targets)
                    # Forward current model
                    outputs = self.model(images)[-1]

                    # Split the batch in old and new classes
                    old_index = targets < self.model.task_offset[t]
                    if sum(old_index):
                        old_images = images[old_index]
                        old_outputs = [outputs[old_index]]
                        pseudo_targets = [old_model(old_images)[-1]]
                    else:
                        old_outputs, pseudo_targets = None, None
                    new_index = targets >= self.model.task_offset[t]
                    if sum(new_index):
                        new_targets = targets[new_index] - \
                            self.model.task_offset[t]
                        new_outputs = [outputs[new_index]]
                    else:
                        new_outputs, new_targets = None, None

                    loss = self.criterion(
                        t, new_outputs, new_targets, old_outputs, pseudo_targets)

                    # Backward
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clipgrad)
                    self.optimizer.step()
                    tot_loss.append(loss.item())
                clock1 = time.process_time()
                train_loss = np.mean(tot_loss)
                print('| Epoch {:3d}, time={:5.3f}s | Train: loss={:.3f} |'.format(
                    e + 1, clock1 - clock0, train_loss), end='')
                self.logger.log_scalar(
                    task=t, iter=e + 1, name="loss", value=train_loss, group="train")

                # Valid
                clock3 = time.process_time()
                valid_loss, valid_acc, _, _, _, _ = self.eval(t, val_loader)
                clock4 = time.process_time()
                print(' Valid: time={:5.3f}s loss={:.3f}, TAw acc={:5.3f}% |'.format(
                    clock4 - clock3, valid_loss, 100 * valid_acc), end='')
                self.logger.log_scalar(
                    task=t, iter=e + 1, name="loss", value=valid_loss, group="valid")
                self.logger.log_scalar(
                    task=t, iter=e + 1, name="acc", value=100 * valid_acc, group="valid")

                # Adapt learning rate - patience scheme - early stopping regularization
                if valid_loss < best_loss:
                    # if the loss goes down, keep it as the best model and end line with a star ( * )
                    best_loss = valid_loss
                    best_model = self.model.get_copy()
                    patience = self.ft_lr_patience
                    print(' *', end='')
                else:
                    # if the loss does not go down, decrease patience
                    patience -= 1
                    if patience <= 0:
                        # if it runs out of patience, reduce the learning rate
                        lr /= self.ft_lr_factor
                        print(' lr={:.1e}'.format(lr), end='')
                        try:
                            if lr < (self.ft_lr_min if t > 0 else self.first_lr_min):
                                # if the lr decreases below minimum, stop the training session
                                print()
                                break
                        except:
                            if lr < self.ft_lr_min:
                                # if the lr decreases below minimum, stop the training session
                                print()
                                break
                        # reset patience and recover best model so far to continue training
                        patience = self.ft_lr_patience
                        self.optimizer.param_groups[0]['lr'] = lr
                        self.model.set_state_dict(best_model)
                self.logger.log_scalar(
                    task=t, iter=e + 1, name="patience", value=patience, group="train")
                self.logger.log_scalar(
                    task=t, iter=e + 1, name="lr", value=lr, group="train")
                print()
            self.model.set_state_dict(best_model)
        return trn_loader

    def post_train_process(self, t, trn_loader, val_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        if t >= self.freeze_after:
            self.model.freeze_backbone()
        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(
            self.model, trn_loader, val_loader.dataset.transform)

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

    def criterion(self, t, outputs, targets, old_outputs=None, old_targets=None):
        """Returns the loss value"""
        loss = 0
        if outputs is not None:
            loss += torch.nn.functional.cross_entropy(
                torch.cat(outputs, dim=1), targets)
        if t > 0 and old_outputs is not None:
            loss += self.lamb * self.gkd(old_outputs, old_targets, t + 1)
        return loss
