import os
import torch
import json
import importlib
import numpy as np
from glob import glob
from torch import optim
from collections import defaultdict
from argparse import ArgumentParser
from pytorch_lightning import LightningModule

appr_module = {
    'anil': 'LightningANIL',
    'maml': 'LightningMAML',
    'matching_net': 'LightningMatchingNetworks',
    'metaoptnet': 'LightningMetaOptNet',
    'proto_net': 'LightningPrototypicalNetworks',
    'relation_net':  'LightningRelationNetworks',
}


class LightningMetaModule(LightningModule):
    
    # Meta-Learning generic parameters
    train_shots = 1
    train_queries = 1
    train_ways = 5
    test_shots = 1
    test_queries = 1
    test_ways = 5
    
    # Optimizer parameters
    lr = 0.001
    scheduler_patience = 10
    scheduler_decay = 0.1
    t0 = 10
    eta_min = 1e-5
    loss_factor = 1
    #save_rec_loss = False
    
    # Suppurt variables
    train_outputs = []
    
    def __init__(self, **kwargs):
        super().__init__()
        # Setting input type for 'split_multimodal()'
        self.num_bytes = kwargs.get("num_bytes", None)
        self.num_pkts = kwargs.get("num_pkts", None)
        self.len_fields = len(kwargs.get("fields", 0))
        self.net_is_mm = kwargs.get("num_modalities", 0) > 0
        self.stage = 0  # Keeps track of the current multimodal stage 
        
        # Meta-Learning generic parameters
        self.train_ways = kwargs.get("train_ways", LightningMetaModule.train_ways)
        self.train_shots = kwargs.get("train_shots", LightningMetaModule.train_shots)
        self.train_queries = kwargs.get("train_queries", LightningMetaModule.train_queries)
        self.test_ways = kwargs.get("test_ways", LightningMetaModule.test_ways)
        self.test_shots = kwargs.get("test_shots", LightningMetaModule.test_shots)
        self.test_queries = kwargs.get("test_queries", LightningMetaModule.test_queries)

        # Optimizer parameters
        self.lr = kwargs.get("lr", LightningMetaModule.lr)
        self.lr_strat = kwargs.get("lr_strat", 'none')
        self.scheduler_decay = kwargs.get(
            "scheduler_decay", LightningMetaModule.scheduler_decay)
        self.scheduler_patience = kwargs.get(
            "scheduler_patience", LightningMetaModule.scheduler_patience)
        self.t0 = kwargs.get("t0", LightningMetaModule.t0)
        self.eta_min = kwargs.get("eta_min", LightningMetaModule.eta_min)
        if self.lr_strat == 'none':
            self.scheduler_patience = float('inf')
            print('No lr scheduler')
        elif self.lr_strat == 'lrop':
            print(f'lrop - patience:{self.scheduler_patience}, decay:{self.scheduler_decay}')
        elif self.lr_strat == 'cawr':
            print(f'cawr - T0:{self.t0}, eta min:{self.eta_min}')
        else:
            raise ValueError('Unsupported lr strategy')
        
        # Loss parameters
        self.loss_factor = kwargs.get("loss_factor", LightningMetaModule.loss_factor)
        self.add_cl = kwargs.get("add_cl", None)
        if self.add_cl is not None:
            print(f'Using {self.add_cl} with factor={self.loss_factor}')
            self.cl_loss = getattr(importlib.import_module(
                'fsl.modules.losses'), self.add_cl)()
        else:
            print('No cluster loss')
        """ if 'Autoencoder' in kwargs.get("network", ''):
            self.rec_loss = torch.nn.MSELoss()
            self.save_rec_loss = True """
        
        # Other
        self._device = kwargs.get("device", "cpu")
        self.setup_val_tasks(kwargs['nc_val'])
        
    @staticmethod
    def factory_approach(approach_name, net, start_from_ckpt, **kwargs):
        Approach = getattr(importlib.import_module(
                name=f'fsl.approach.{approach_name}'), appr_module[approach_name])
        
        if not start_from_ckpt:
            # Create a new approach
            return 0, Approach(net=net, **kwargs)

        # Else restore an approach from checkpoint
        num_modalities = kwargs.get("num_modalities", 0)
        recover_path = os.path.split(kwargs.get("recover_args_path", ''))[0]
        last_recover_path = recover_path
         
        for mod in range(1, num_modalities + 1):
            mod_recover_path = recover_path.replace('stage_0', f'stage_{mod}')
            if not os.path.exists(mod_recover_path):      
                ckpt_path = glob(os.path.join(last_recover_path, 'checkpoints', '*'))[0]
                print(f'Recovering from checkpoint {ckpt_path}')
                return mod, Approach.load_from_checkpoint(ckpt_path, net=net, **kwargs)
            else:
                last_recover_path = mod_recover_path

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(
            parents=[parent_parser],
            add_help=True,
            conflict_handler="resolve",
        )
        parser.add_argument(
            "--train_ways", type=int, default=LightningMetaModule.train_ways
        )
        parser.add_argument(
            "--train_shots", type=int, default=LightningMetaModule.train_shots
        )
        parser.add_argument(
            "--train_queries", type=int, default=LightningMetaModule.train_queries
        )
        parser.add_argument(
            "--test_ways", type=int, default=LightningMetaModule.test_ways
        )
        parser.add_argument(
            "--test_shots", type=int, default=LightningMetaModule.test_shots
        )
        parser.add_argument(
            "--test_queries", type=int, default=LightningMetaModule.test_queries
        )
        parser.add_argument(
            "--lr", type=float, default=LightningMetaModule.lr
        )
        parser.add_argument(
            '--lr-strat', type=str, nargs='?', choices=['lrop', 'cawr', 'none'],
            default='none'
        )
        parser.add_argument(
            "--scheduler_patience", type=int, default=LightningMetaModule.scheduler_patience,
        )
        parser.add_argument(
            "--scheduler_decay", type=float, default=LightningMetaModule.scheduler_decay
        )
        parser.add_argument(
            "--t0", type=int, default=LightningMetaModule.t0
        )
        parser.add_argument(
            "--eta-min", type=float, default=LightningMetaModule.eta_min
        )
        parser.add_argument(
            '--add-cl', type=str, nargs='?', choices=['DaviesBouldinLoss', 'SilhouetteLoss'],
            default=None
        )
        parser.add_argument(
            '--loss-factor', type=float, default=LightningMetaModule.loss_factor
        )
        return parser

    ####
    ## ON STEP CALLBACKS
    ####
    
    def training_step(self, batch, batch_idx):
        outputs = self.meta_learn(
            batch, batch_idx, self.train_ways, self.train_shots, self.train_queries
        )
        outputs['loss'] = self.add_cluster_loss(
            outputs['loss'], outputs['query'], outputs['query_labels']
        )  # Add a cluster-based loss to the classification loss if add_cl is True
        self.log(
            "train_loss",
            outputs['loss'].item(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "train_accuracy",
            outputs['accuracy'].item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.train_outputs.append(outputs)
        return outputs['loss']

    def validation_step(self, batch, batch_idx):
        outputs = self.meta_learn(
            batch, batch_idx, self.val_ways, self.val_shots, self.val_queries
        )
        outputs['loss'] = self.add_cluster_loss(
            outputs['loss'], outputs['query'], outputs['query_labels']
        )  # Add a cluster-based loss to the classification loss if add_cl is True
        self.log(
            "valid_loss",
            outputs['loss'].item(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "valid_accuracy",
            outputs['accuracy'].item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return outputs

    def test_step(self, batch, batch_idx):
        outputs = self.meta_learn(
            batch, batch_idx, self.test_ways, self.test_shots, self.test_queries
        )
        outputs['loss'] = self.add_cluster_loss(
            outputs['loss'], outputs['query'], outputs['query_labels']
        )  # Add a cluster-based loss to the classification loss if add_cl is True
        self.log(
            "test_loss",
            outputs['loss'].item(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "test_accuracy",
            outputs['accuracy'].item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return outputs

    ####
    ## ON EPOCH END CALLBACKS
    ####
    
    def training_epoch_end(self, _):
        self._save_outputs('train_data', self.train_outputs)
        self.train_outputs = []

    def validation_epoch_end(self, outputs):
        self._save_outputs('val_data', outputs)
        
    def test_epoch_end(self, outputs):
        self._save_outputs('test_data', outputs)

    def _save_outputs(self, folder_name, outputs):
        save_path = f'{self.logger.log_dir}/{folder_name}/'
        os.makedirs(save_path, exist_ok=True)
        
        save_outs = defaultdict(list)
        for output in outputs:
            for key in output.keys():
                save_outs[key].append(output[key].detach().cpu().numpy() 
                                      if key != 'le_map' else output[key])

        e = self.trainer.current_epoch
        self._save_branch_logits(save_path, save_outs, e)
        np.savez_compressed(f'{save_path}logits_ep{e}.npz',
                            logits=save_outs['logits'])
        np.savez_compressed(f'{save_path}supports_ep{e}.npz',
                            supports=save_outs['support'])
        np.savez_compressed(f'{save_path}queries_ep{e}.npz',
                            queries=save_outs['query'])
        np.savez_compressed(f'{save_path}labels_ep{e}.npz',
                            query_labels=save_outs['query_labels'], 
                            support_labels=save_outs['support_labels'])
        """ if self.save_rec_loss: # Save also the rec loss if the backbone is an AE
            np.savez_compressed(
                f'{save_path}losses_ep{e}.npz',
                loss=save_outs['loss'], rec_loss=save_outs['rec_loss'])
        else: """
        np.savez_compressed(
            f'{save_path}losses_ep{e}.npz', loss=save_outs['loss'])
        with open(f'{save_path}le_ep{e}.json', 'w') as f:
            json.dump(save_outs['le_map'], f)

    ####
    ## OPTIMIZER SETUP
    ####
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        if self.lr_strat == 'cawr':
            lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=self.t0, T_mult=1, eta_min=self.eta_min
            )
            return [optimizer], [lr_scheduler]
        elif self.lr_strat in ['none', 'lrop']:
            # If it is 'none' the patience is inf
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, patience=self.scheduler_patience, mode='min',
                        factor=self.scheduler_decay, verbose=True
                    ),
                    'interval': 'epoch',  # The unit of the scheduler's step size
                    'monitor': 'valid_loss',  # Metric to monitor
                    'frequency': 1,  # How many epochs/steps should pass between calls to `scheduler.step()`
                    'name': 'ReduceLROnPlateau'  # Needed for logging
                }}
        else:
            raise ValueError('Unsupported lr strategy')
    
    ####
    ## UTILITY FUNCTIONS
    ####

    def label_encoding(self, labels):
        # Label encoding from 0 to Ways
        le = dict(zip(labels.unique().tolist(), range(labels.unique().size(0))))
        labels = torch.tensor([le[int(label)]
                              for label in labels], device=self._device)
        return labels, le

    def split_multimodal(self, x, batch_dim):
        """
        When mimetic is used, an x has as shape: [1,num_bytes+num_pckt*fields].
        This function will split the x in a list of 2 elements, one for each modality.
        """
        if not self.net_is_mm:
            return x
        x_load, x_fields = torch.split(x, self.num_bytes, dim=1)
        x_load = x_load.unsqueeze(dim=1)  # Wang input type (e.g. [1,576])
        # Lopez input type (e.g. [1,20,4])
        x_fields = x_fields.view(batch_dim, 1, self.num_pkts, self.len_fields)
        return [x_load, x_fields]

    def setup_val_tasks(self, nc_val):
        # Setting N and K for meta-validation
        self.val_ways = self.test_ways if nc_val != 0 else self.train_ways
        self.val_shots = self.test_shots if nc_val != 0 else self.train_shots
        self.val_queries = self.test_queries if nc_val != 0 else self.train_queries

    def add_cluster_loss(self, loss, feature, label):
        if self.add_cl is None:
            return loss
        return self.loss_factor*self.cl_loss(feature, label) + loss

    def _save_branch_logits(self, save_path, save_outs, e):
        if self.stage == 2 and 'test' in save_path and 'logits_' in save_outs:
            bl_path = f'{self.logger.log_dir}/branches_logits/'
            os.makedirs(bl_path, exist_ok=True)
            np.savez_compressed(
                f'{bl_path}logits_ep{e}.npz',
                logits_cnn=save_outs['logits_cnn'],
                logits_gru=save_outs['logits_gru']
            )
