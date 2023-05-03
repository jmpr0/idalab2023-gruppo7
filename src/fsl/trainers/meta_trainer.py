import json
from glob import glob

import fsl.util.rng
import fsl.util.callbacks
from fsl.trainers.interface_trainer import PLTrainer 


class MetaTrainer(PLTrainer):
    """
    Adds functionalities for meta-learning training procedure and for Meta-Mimetic
    """
    def __init__(self, args, num_modalities=0):
        super().__init__(args)
        self.stage = 0
        self.num_modalities = num_modalities
        self.log_path = args.default_root_dir
        
        if self.num_modalities != 0:
            self.args.default_root_dir = f'{self.log_path}_stage_{self.stage}'
            # Setting up epochs for MIMETIC
            pt_e = int(0.28*args.max_epochs)
            ft_e = args.max_epochs - 2*pt_e
            self.metamimetic_epochs = [pt_e, pt_e, ft_e]
            self.args.max_epochs = self.metamimetic_epochs[0]
            print(f'Multi-modal epochs: {self.metamimetic_epochs}')
        
        self._setup_first_trainer()
        
        
    def fit(self, approach, datamodule=None, train_dataloader=None, val_dataloader=None):
        """ Wraps Pytorch Lightning 'fit()' """
        self._swap_modality(approach)
        self._set_rng_states()
        
        self.trainers[self.stage].fit(
            model=approach,
            datamodule=datamodule,
            train_dataloader=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        self.tick()
        
    def test(self):
        """ Wraps Pytorch Lightning 'test()' """
        best_model_path = glob(f'{self.trainers[-1].logger.log_dir}/checkpoints/*')[0]
        eval_res = self.trainers[-1].test(ckpt_path=best_model_path) # Meta-Testing
        return eval_res

    def tick(self):
        # If it's the last stage or the model is single-modal return
        if self.stage == self.num_modalities or self.num_modalities == 0:
            return

        # Setups a new trainer for the next phase
        self.stage += 1
        self.args.default_root_dir = f'{self.log_path}_stage_{self.stage}'
        self.args.max_epochs = self.metamimetic_epochs[self.stage]
        self.callbacks.append(self.create_callbacks())
        self.trainers.append(self.create_trainer())
        print('-'*80)
        print(f'New max epoch: {self.args.max_epochs}')
        
    def create_callbacks(self, callbacks=[]):
        callbacks = []
        callbacks.append(fsl.util.callbacks.TrackTestAccuracyCallback())
        return super().create_callbacks(callbacks)
        
    def save_results(self, eval_res):
        # Saving usefull info
        with open(f'{self.trainers[-1].logger.log_dir}/test_results.json', 'w') as f:
            json.dump(eval_res, f)
        fsl.util.per_epoch_logger.plot_experiment_log(self.trainers[-1].logger.log_dir)
        return self.trainers[-1].logger.log_dir

    def _swap_modality(self, approach):
        if self.num_modalities == 0:
            return

        approach.stage = self.stage  # Setup PL module for a new task
        # If it's the last stage, freeze the single-modality backbones
        if self.stage == self.num_modalities:
            approach.net.freeze_mod_backbone()
                          
    def _set_rng_states(self):
        if self.stage != 0:
            fsl.util.rng.seed_everything(self.args.seed + self.stage)