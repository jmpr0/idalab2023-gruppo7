import importlib
from copy import deepcopy
from argparse import ArgumentParser
from pytorch_lightning import Trainer

import fsl.util.callbacks
import fsl.util.rng
from datasets.data_loader import get_tl_loaders
from networks.network import LLL_Net
from fsl.trainers.tl_trainer import TLTrainer
from fsl.approach import (
    LightningTLModule,
    LightningFineTuning,
    LightningFreezing,
    PreTraining # TODO: not working, adaptation required
)

def main():
    ####
    ## 0 - PARSING INPUT
    ####    
    parser = ArgumentParser(conflict_handler="resolve", add_help=True) 
    parser = fsl.util.callbacks.EarlyStoppingDoubleMetric.add_model_specific_args(parser)
    parser = LightningFineTuning.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--approach", type=str, default="finetuning")
    parser.add_argument("--pt-only", action='store_true', default=False)
    # Dataset args
    parser.add_argument("--dataset", type=str, default='iot23')
    parser.add_argument('--num_bytes', type=int, default=None)
    parser.add_argument('--num_pkts', type=int, default=None)
    parser.add_argument("--num_tasks", type=int, default=100)
    parser.add_argument('--nc_pretrain', type=int, default=None)
    parser.add_argument(
        '--fields', type=str, default=[], 
        choices=['PL', 'IAT', 'DIR', 'WIN', 'FLG', 'TTL'],
        help='Field or fields used (default=%(default)s)',
        nargs='+', metavar='FIELD'
    )
    # Model args
    parser.add_argument("--network", type=str, default="Lopez17CNN")
    parser.add_argument("--weights_path", type=str, default=None) # TODO: da testare
    parser.add_argument("--out-features-size", type=int, default=None)
    parser.add_argument("--adaptation-epochs", type=int, default=50)
    parser.add_argument(
        "--scale", type=float, default=1,
        help='Scaling factor to modify the number of trainable '
        'parameters used by model (default=%(default)s)'
    )
    parser.add_argument(
        '--keep-existing-head', action='store_true',
        help='Disable removing classifier last layer (default=%(default)s)'
    )
    args = parser.parse_args()
    dict_args = vars(args)
    
    dict_args_copy = deepcopy(dict_args)  # Used to store the initial args before the training procedure
    dict_args_copy.pop('tpu_cores')  # Removing the tpu_cores entry that should not be saved

    try:
        dict_args['device'] = 'cuda' if args.gpus > 0 else 'cpu'
    except:
        dict_args['device'] = 'cpu'
    
    ####
    ## 1 - GET LOADERS
    ####  
    fsl.util.rng.seed_everything(args.seed)
    
    ways, pretrain_datamodule, finetune_taskset = get_tl_loaders(
        dataset=args.dataset,
        num_bytes=args.num_bytes, 
        num_pkts=args.num_pkts, 
        fields=args.fields, 
        seed=args.seed, 
        nc_pretrain=args.nc_pretrain,
        pt_only=args.pt_only,
        queries=args.queries, 
        shots=args.shots, 
        num_tasks=args.num_tasks
    )

    ####
    ## 2 - GET MODEL AND ALGORITHM
    ####
    net = getattr(importlib.import_module(name='networks'), args.network)
    init_model = net(
        num_bytes=args.num_bytes,
        num_pkts=args.num_pkts,
        num_fields=len(args.fields),
        out_features_size=args.out_features_size,
        scale=args.scale
    )
    # TODO usare il factory method come in main_meta.py e implementare il train multimodale
    net = LLL_Net(
        init_model,
        remove_existing_head=not args.keep_existing_head,
    )
    for way in ways:
        net.add_head(num_outputs=way)
    print(net)

    if args.patience == -1:
        args.patience = float('inf')
        
    approach = LightningTLModule.factory_approach(
        args.approach, net, False, **dict_args)

    ####
    ## 3 - TRAIN AND TEST
    ####
    tl_trainer = TLTrainer(args)
    tl_trainer.save_args(dict_args_copy)
    
    # Pre-Training fit
    tl_trainer.fit(approach=approach, datamodule=pretrain_datamodule) 
    # Pre-Training test
    eval_res = tl_trainer.test() 
    if not args.pt_only:
        # FSL-like fine-tuning
        ft_res = tl_trainer.fine_tune(approach=approach, dataloader=finetune_taskset) 
        eval_res = {**eval_res[0], **ft_res}
    tl_trainer.save_results(eval_res)
    # TODO: implement final logger


if __name__ == '__main__':
    main()