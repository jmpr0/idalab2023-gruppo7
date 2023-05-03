import os
import torch
from argparse import ArgumentParser
from torch import nn
from torch import optim
from pytorch_lightning import LightningModule


class PreTraining(LightningModule):

    lr = 0.001
    scheduler_patience = 10
    scheduler_decay = 0.1
    saved_weights_path = "../saved_weights/" 

    def __init__(self, model, loss=None, **kwargs):
        super().__init__()
        # TODO: add lr_strat
        self.lr = kwargs.get("lr", PreTraining.lr)
        self.scheduler_decay = kwargs.get("scheduler_decay", PreTraining.scheduler_decay)
        lr_patience = kwargs.get("scheduler_patience", PreTraining.scheduler_patience)
        self.scheduler_patience = float('inf') if lr_patience == -1 else lr_patience
        self.saved_weights_path = kwargs.get("saved_weights_path", PreTraining.saved_weights_path)
        self.net_name =  kwargs.get("network", 'model')
        print('No lr scheduler') if lr_patience == -1 \
        else print(f'lr scheduler - patience:{self.scheduler_patience} - factor:{self.scheduler_decay}')

        self.loss = loss or nn.MSELoss()
        self.model = model
        self._device = kwargs.get("device", "cpu")


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(
            parents=[parent_parser],
            add_help=True,
            conflict_handler="resolve",
        )
        parser.add_argument(
            "--lr", type=float, default=PreTraining.lr
        )
        parser.add_argument(
            "--scheduler_patience", type=int, default=PreTraining.scheduler_patience,
            help='inf if set equal to -1'
        )
        parser.add_argument(
            "--scheduler_decay", type=float, default=PreTraining.scheduler_decay
        )
        parser.add_argument(
            "--saved_weights_path", type=str, default=PreTraining.saved_weights_path
        )
        return parser

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
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
            }
        }

    def _get_reconstruction_error(self, batch):
        x, _ = batch
        _, _, x_bar = self.model(x)
        x = torch.reshape(x, x_bar.shape)
        x = torch.sigmoid(x) if self.net_name == 'ShallowAutoencoder' else x
        rec_loss = self.loss(x, x_bar)
        return rec_loss

    def training_step(self, batch, batch_idx):
        # TODO: do this only if net is an AE
        loss = self._get_reconstruction_error(batch)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_error(batch)
        self.log(
            "valid_loss",
            loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

    def training_epoch_end(self, _):
        os.makedirs(self.saved_weights_path, exist_ok=True)
        """ for name, param in self.model.model.named_parameters():
            print(name, param.requires_grad, torch.sum(param).item()) """
        torch.save(
            self.model.model.state_dict(),
            f'{self.saved_weights_path}{self.net_name}.pt')
