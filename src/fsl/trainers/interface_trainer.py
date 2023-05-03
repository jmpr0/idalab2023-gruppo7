import os
import json
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from abc import ABC, abstractmethod

import fsl.util.callbacks
import fsl.util.per_epoch_logger

class PLTrainer(ABC):
    """
    Wrapper for Pytorch Lightning trainer for handling different FSL paradigm 
    """
    def __init__(self, args):
        self.args = args
        
    def _setup_first_trainer(self):
        self.callbacks = [self.create_callbacks()]
        self.trainers = [self.create_trainer()]
        
    def create_trainer(self):
        """ Create a PL Trainer given args dict """
        return pl.Trainer.from_argparse_args(
            self.args,
            gpus=self.args.gpus,
            deterministic=True,
            accumulate_grad_batches=self.args.accumulate_grad_batches,
            checkpoint_callback=ModelCheckpoint(mode=self.args.mode, monitor=self.args.monitor),
            callbacks=self.callbacks[-1]
        )
        
    def create_callbacks(self, callbacks=[]):
        """ Create a set of default callbacks with the one passed in 'callbacks' """
        default_callbacks = [
            fsl.util.callbacks.NoLeaveProgressBar(),
            fsl.util.callbacks.LearningRateMonitorOnLog(logging_interval='epoch'),
            fsl.util.callbacks.EarlyStoppingDoubleMetric(
                monitor=self.args.monitor, 
                min_delta=self.args.min_delta,
                patience=self.args.patience, 
                mode=self.args.mode, 
                verbose=True,
                double_monitor=self.args.double_monitor)
        ]
        return default_callbacks + callbacks
    
    def save_args(self, dict_args):
        """ Storing initial argument dict for each modality """
        os.makedirs(f'{self.trainers[-1].logger.log_dir}', exist_ok=True)
        with open(f'{self.trainers[-1].logger.log_dir}/dict_args.json', 'w') as f:
            json.dump(dict_args, f)
    
    @abstractmethod
    def fit(self):
        pass
    
    @abstractmethod
    def test(self):
        pass
    
    @abstractmethod
    def save_results(self):
        pass