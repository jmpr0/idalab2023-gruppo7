import importlib
import json
from copy import deepcopy
from argparse import ArgumentParser
from glob import glob
from pytorch_lightning import Trainer

import fsl.util.callbacks
import fsl.util.cleanup
import fsl.util.rng
from networks.interface_network import INetwork
from datasets.data_loader import get_episodic_datamodule
from fsl.modules.fce import Fce
from fsl.trainers.meta_trainer import MetaTrainer
from fsl.approach import (
    LightningMetaModule,
    LightningANIL,
    LightningMAML,
    LightningMatchingNetworks,
    LightningMetaOptNet,
    LightningPrototypicalNetworks,
    LightningRelationNetworks,
)

def main(): 
    ####
    ## 0 - PARSING INPUT
    ####    
    parser = ArgumentParser(conflict_handler="resolve", add_help=True)
    parser = LightningPrototypicalNetworks.add_model_specific_args(parser)
    parser = LightningMAML.add_model_specific_args(parser)
    parser = LightningMetaOptNet.add_model_specific_args(parser)
    parser = LightningANIL.add_model_specific_args(parser)
    parser = LightningMatchingNetworks.add_model_specific_args(parser)
    parser = LightningRelationNetworks.add_model_specific_args(parser)
    parser = Fce.add_model_specific_args(parser)
    parser = fsl.util.callbacks.EarlyStoppingDoubleMetric.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    parser.add_argument('--recover-args-path', type=str, default=None)
    parser.add_argument("--approach", type=str, default="protonet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cleanup", action='store_true', default=False)
    # Dataset args
    parser.add_argument("--dataset", type=str, default='iot23')
    parser.add_argument("--acc_grad_batches", type=int, default=1)
    parser.add_argument("--meta-batches-per-epoch", type=int, default=10)
    parser.add_argument("--num_tasks", type=int, default=-1)
    parser.add_argument('--num_bytes', type=int, default=None)
    parser.add_argument('--num_pkts', type=int, default=None)
    parser.add_argument("--nc-train", type=int, default=None)
    parser.add_argument("--nc-val", type=int, default=None)
    parser.add_argument("--nc-test", type=int, default=None)
    parser.add_argument("--augs", type=str, default='')
    parser.add_argument("--hold-out", action='store_true', default=False)
    parser.add_argument(
        '--fields', type=str, default=[],
        choices=['PL', 'IAT', 'DIR', 'WIN', 'FLG', 'TTL'],
        help='Field or fields used (default=%(default)s)',
        nargs='+', metavar='FIELD'
    )
    # Model args
    parser.add_argument("--network", type=str, default="Lopez17CNN")
    parser.add_argument("--out-features-size", type=int, default=None)
    parser.add_argument("--weights_path", type=str, default=None)
    parser.add_argument("--return_concat", action='store_true', default=False)
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

    start_from_ckpt = False
    if args.recover_args_path is not None:
        # Restore saved parameters
        with open(args.recover_args_path) as f:
            recovered_dict_args = json.load(f)
        for k, v in recovered_dict_args.items():
            if k != 'recover_args_path':
                dict_args[k] = v
        start_from_ckpt = True

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

    episodic_datamodule = get_episodic_datamodule(
        dataset=args.dataset,
        num_tasks=args.num_tasks,
        train_shots=args.train_shots,
        train_ways=args.train_ways,
        train_queries=args.train_queries,
        test_shots=args.test_shots,
        test_ways=args.test_ways,
        test_queries=args.test_queries,
        acc_grad_batches=args.acc_grad_batches,
        meta_batches_per_epoch=args.meta_batches_per_epoch,
        num_bytes=args.num_bytes,
        num_pkts=args.num_pkts,
        fields=args.fields,
        seed=args.seed,
        nc_train=args.nc_train,
        nc_val=args.nc_val,
        nc_test=args.nc_test,
        augs=args.augs,
        hold_out=args.hold_out
    )
    
    ####
    ## 2 - GET MODEL AND ALGORITHM
    ####
    net = getattr(importlib.import_module(name='networks'), args.network)
    init_model = net(
        pretrained=False,
        num_bytes=args.num_bytes,
        num_pkts=args.num_pkts,
        num_fields=len(args.fields),
        out_features_size=args.out_features_size,
        scale=args.scale
    )
    net, num_modalities = INetwork.factory_net(
        init_model, 
        remove_existing_head=not args.keep_existing_head, 
        weights_path=args.weights_path
    )
    net.add_head(
        num_outputs=args.train_ways, 
        concat=args.return_concat if num_modalities > 0 else False
    )

    # Skip early stopping if nc_val is 0 or if patience is et to -1
    if args.nc_val == 0 or args.patience == -1: 
        args.patience = float('inf') 
    
    dict_args['fce'] = Fce(net.out_size, **dict_args) if dict_args['apply_fce'] else None
    dict_args['num_modalities'] = num_modalities
    
    rec_mod, approach = LightningMetaModule.factory_approach(
        args.approach, net, start_from_ckpt, **dict_args) 
    
    ####
    ## 3 - TRAIN AND TEST
    ####
    meta_trainer = MetaTrainer(args, num_modalities)
    # Train each modality
    for mod in range(num_modalities + 1): 
        # Skip modalities already done
        if mod < rec_mod:
            meta_trainer.tick()
            continue
        # Save parameters
        meta_trainer.save_args(dict_args_copy)
        # Meta-training
        meta_trainer.fit(approach=approach, datamodule=episodic_datamodule)
    # Meta-Testing
    eval_res = meta_trainer.test() 
    
    log_dir = meta_trainer.save_results(eval_res)
    if args.cleanup:
        fsl.util.cleanup.clean_files(log_dir) # Remove unnecessary files


if __name__ == '__main__':
    main()