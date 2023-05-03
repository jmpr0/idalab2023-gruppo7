import json
import numpy as np
from tqdm import tqdm
from glob import glob
from fsl.trainers.interface_trainer import PLTrainer


class TLTrainer(PLTrainer):
    """
    Adds functionalities for transfer-learning training procedure
    """
    def __init__(self, args):
        super().__init__(args)
        self._setup_first_trainer()
        
        
    def fit(self, approach, datamodule=None, train_dataloader=None, val_dataloader=None):
        self.trainers[0].fit(
            model=approach,
            datamodule=datamodule,
            train_dataloader=train_dataloader,
            val_dataloaders=val_dataloader,
        )
    
    def test(self):  
        """ Wraps Pytorch Lightning 'test()' """
        best_model_path = glob(f'{self.trainers[-1].logger.log_dir}/checkpoints/*')[0]
        eval_res = self.trainers[0].test(ckpt_path=best_model_path)
        return eval_res  
    
    def fine_tune(self, approach, dataloader):
        """ Implementation of the fine-tuning stage with a FSL procedure """
        # Resuming best weights
        best_ckpt_path = glob(f'{self.trainers[-1].logger.log_dir}/checkpoints/*')[0]
        print(f'Resuming {best_ckpt_path} for fine-tuning')
        print('-'*80)
        best_approach = type(approach).load_from_checkpoint(
            net=approach.net, checkpoint_path=best_ckpt_path, **vars(self.args))

        # Adaptation on the Support Set and evaluation on the Query Set
        losses = []
        accuracies = []
        outputs = []
        ft_loop = tqdm(dataloader)
        
        for batch in ft_loop:
            ft_net = None
            # Fine-tune for N epochs
            for _ in tqdm(range(self.args.adaptation_epochs), leave=False):
                output, ft_net = best_approach.adaptation_step(batch, ft_net) 
            
            # Collect metrics from the last epoch 
            loss = output['loss'].item()
            acc = output['accuracy'].item()
            ft_loop.set_postfix(loss=loss, acc=acc)
            losses.append(loss)
            accuracies.append(acc)
            outputs.append(output)
        
        # Return the avg per episode metrics     
        eval_res = {
            'fine_tuning loss': np.array(losses).mean(),
            'fine_tuning accuracy': np.array(accuracies).mean()
        }
        best_approach.adaptation_epoch_end(outputs, f'{self.trainers[-1].logger.log_dir}')
        print('\n'+'-'*80)
        print(eval_res)
        return eval_res
    
    def save_results(self, eval_res):
        # Saving usefull info
        with open(f'{self.trainers[-1].logger.log_dir}/test_results.json', 'w') as f:
            json.dump(eval_res, f)