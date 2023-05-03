import argparse
import importlib
import os
import time
from functools import reduce

import git
import numpy as np
import pandas as pd
import torch

import approach
import utils
from datasets.data_loader import get_loaders
from datasets.dataset_config import dataset_config
from last_layer_analysis import last_layer_analysis
from loggers.exp_logger import MultiLogger
from networks import tvmodels, netmodels, nnmodels, set_tvmodel_head_var
from ml_models import mlmodels
from approach import ova_mg_approaches


# TODO: enhance the timing and versioning
# TODO: enhance the gridsearch

def main(argv=None):
    tstart = time.time()
    # Arguments
    parser = argparse.ArgumentParser(description='FACIL - Framework for Analysis of Class Incremental Learning')

    # miscellaneous args
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU (default=%(default)s)')
    parser.add_argument('--results-path', type=str, default='../results',
                        help='Results path (default=%(default)s)')
    parser.add_argument('--exp-name', default=None, type=str,
                        help='Experiment name (default=%(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default=%(default)s)')
    parser.add_argument('--log', default=['disk'], type=str, choices=['disk', 'tensorboard'],
                        help='Loggers used (disk, tensorboard) (default=%(default)s)', nargs='*', metavar='LOGGER')
    parser.add_argument('--save-models', action='store_true',
                        help='Save trained models (default=%(default)s)')
    parser.add_argument('--load-base-model', action='store_true',
                        help='Retrieve the stored base model (default=%(default)s)')
    parser.add_argument('--base-model-path', type=str, default=None,
                        help='Base model path (default=%(default)s)')
    parser.add_argument('--last-layer-analysis', action='store_true',
                        help='Plot last layer analysis (default=%(default)s)')
    parser.add_argument('--no-cudnn-deterministic', action='store_true',
                        help='Disable CUDNN deterministic (default=%(default)s)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable the debug execution of the script (default=%(default)s)')
    # dataset args
    parser.add_argument('--datasets', default=['cifar100'], type=str, choices=list(dataset_config.keys()),
                        help='Dataset or datasets used (default=%(default)s)', nargs='+', metavar='DATASET')
    parser.add_argument('--num-workers', default=1, type=int, required=False,
                        help='Number of subprocesses to use for dataloader (default=%(default)s)')
    parser.add_argument('--pin-memory', default=False, type=bool, required=False,
                        help='Copy Tensors into CUDA pinned memory before returning them (default=%(default)s)')
    parser.add_argument('--batch-size', default=64, type=int, required=False,
                        help='Number of samples per batch to load (default=%(default)s)')
    parser.add_argument('--num-tasks', default=4, type=int, required=False,
                        help='Number of tasks per dataset (default=%(default)s)')
    parser.add_argument('--nc-first-task', default=0, type=int, required=False,
                        help='Number of classes of the first task (default=%(default)s)')
    parser.add_argument('--nc-incr-tasks', default=0, type=int, required=False,
                        help='Number of classes of the incremental tasks (default=%(default)s)')
    parser.add_argument('--use-valid-only', action='store_true',
                        help='Use validation split instead of test (default=%(default)s)')
    parser.add_argument('--stop-at-task', default=0, type=int, required=False,
                        help='Stop training after specified task (default=%(default)s)')
    parser.add_argument('--num-bytes', default=None, type=int,
                        help='Number of bytes to select from the dataset')
    parser.add_argument('--num-pkts', default=None, type=int,
                        help='Number of packets to select from the dataset')
    parser.add_argument('--fields', default=[], type=str, choices=['PL', 'IAT', 'DIR', 'WIN', 'FLG', 'PL_DIR'],
                        help='Field or fields used (default=%(default)s)', nargs='+', metavar='FIELD')
    parser.add_argument('--last-class', default=-1, type=int,
                        help='Last target class. When different from -1, the last class is blocked and its value is '
                             'added to the seed for previous classes randomization (default=%(default)s)')
    # model args
    parser.add_argument('--network', default=None, type=str, choices=nnmodels,
                        help='Network architecture used (default=%(default)s)', metavar='NETWORK')
    parser.add_argument('--ml-model', default=None, type=str, choices=mlmodels,
                        help='Machine learning algorithm used (default=%(default)s)', metavar='ML_MODEL')
    parser.add_argument('--out-features-size', default=None, type=int,
                        help='Feature vector size (default=%(default)s)')
    parser.add_argument('--keep-existing-head', action='store_true',
                        help='Disable removing classifier last layer (default=%(default)s)')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained backbone (default=%(default)s)')
    parser.add_argument('--pre-allocated-output', action='store_true',
                        help='If the model output should be entirely pre-allocated, viz. non-incremental')
    # training args
    parser.add_argument('--approach', default='finetuning', type=str, choices=approach.__all__,
                        help='Learning approach used (default=%(default)s)', metavar='APPROACH')
    parser.add_argument('--nepochs', default=200, type=int, required=False,
                        help='Number of epochs per training session (default=%(default)s)')
    parser.add_argument('--modspec-nepochs', default=200, type=int, required=False,
                        help='Number of epochs per modality-specific training session (default=%(default)s)')
    parser.add_argument('--lr', default=0.1, type=float, required=False,
                        help='Starting learning rate (default=%(default)s)')
    parser.add_argument('--lr-min', default=1e-4, type=float, required=False,
                        help='Minimum learning rate (default=%(default)s)')
    parser.add_argument('--lr-factor', default=3, type=float, required=False,
                        help='Learning rate decreasing factor (default=%(default)s)')
    parser.add_argument('--first-lr', default=0.1, type=float, required=False,
                        help='Starting learning rate for the first task (default=%(default)s)')
    parser.add_argument('--first-lr-min', default=1e-4, type=float, required=False,
                        help='Minimum learning rate for the first task (default=%(default)s)')
    parser.add_argument('--first-lr-factor', default=3, type=float, required=False,
                        help='Learning rate decreasing factor for the first task (default=%(default)s)')
    parser.add_argument('--lr-patience', default=5, type=int, required=False,
                        help='Maximum patience to wait before decreasing learning rate (default=%(default)s)')
    parser.add_argument('--clipping', default=10000, type=float, required=False,
                        help='Clip gradient norm (default=%(default)s)')
    parser.add_argument('--momentum', default=0.0, type=float, required=False,
                        help='Momentum factor (default=%(default)s)')
    parser.add_argument('--first-momentum', default=0.0, type=float, required=False,
                        help='Momentum factor for the first task (default=%(default)s)')
    parser.add_argument('--weight-decay', default=0.0, type=float, required=False,
                        help='Weight decay (L2 penalty) (default=%(default)s)')
    parser.add_argument('--warmup-nepochs', default=0, type=int, required=False,
                        help='Number of warm-up epochs (default=%(default)s)')
    parser.add_argument('--warmup-lr-factor', default=1.0, type=float, required=False,
                        help='Warm-up learning rate factor (default=%(default)s)')
    parser.add_argument('--multi-softmax', action='store_true',
                        help='Apply separate softmax for each task (default=%(default)s)')
    parser.add_argument('--fix-bn', action='store_true',
                        help='Fix batch normalization after first task (default=%(default)s)')
    parser.add_argument('--eval-on-train', action='store_true',
                        help='Show train loss and accuracy (default=%(default)s)')
    # gridsearch args
    parser.add_argument('--gridsearch-tasks', default=-1, type=int,
                        help='Number of tasks to apply GridSearch (-1: all tasks) (default=%(default)s)')
    parser.add_argument('--validation', default=0.1, type=float, required=False,
                        help='Validation fraction (default=%(default)s)')
    parser.add_argument('--timestamp-to-recover', default=None, type=int,
                        help='Timestamp to try the recovery of experiment (default=%(default)s).')
    parser.add_argument('--shots', default=-1, type=int,
                        help='Number of samples for each class (shots) (-1: all samples) (default=%(default)s)')
    parser.add_argument('--fseed', default=-1, type=int,
                        help='Random seed for downsampling (default=%(default)s)')
    parser.add_argument('--scale', default=1, type=float,
                        help='Random seed for downsampling (default=%(default)s)')
    parser.add_argument('--predict-tasks', action='store_true',
                        help='Defines an incremental model predicting tasks, not classes (default=%(default)s)')
    parser.add_argument('--statistical', action='store_true',
                        help='Use statistical dataset instead of normal one  (default=%(default)s)') ## TODO: attemp for statistical features gate
    # Args -- Incremental Learning Framework
    args, extra_args = parser.parse_known_args(argv)

    assert args.seed == 0 if 'appclassnet' in args.datasets else True

    if args.timestamp_to_recover is not None:
        from glob import glob
        import json

        args_dict_fn = glob('%s/**/args-%d.txt' % (args.results_path, args.timestamp_to_recover), recursive=True)

        assert len(args_dict_fn) == 1, '%s dict(s) found for timestamp %s' % (
            len(args_dict_fn), args.timestamp_to_recover)

        with open(args_dict_fn[0]) as fin:
            args_dict = json.loads(fin.read())

        # Overriding external passable parameters for recovering
        args_dict['debug'] = args.debug
        args_dict['results_path'] = args.results_path
        args_dict['gpu'] = args.gpu
        args_dict['timestamp_to_recover'] = args.timestamp_to_recover
        for arg in vars(args):
            setattr(args, arg, args_dict.pop(arg, None))

        for k, v in args_dict.items():
            extra_args.append('--%s' % k.replace('_', '-'))
            if isinstance(v, list):
                for v0 in v:
                    extra_args.append(str(v0))
            elif str(v).lower() not in ['false', 'true']:
                extra_args.append(str(v))
        print(args, extra_args)

    if args.debug:
        setattr(args, 'results_path', args.results_path.rstrip('/') + '_debug')

    if args.nc_incr_tasks > 0:
        args.num_tasks = args.stop_at_task
    
    is_ml = args.ml_model is not None

    args.results_path = os.path.expanduser(args.results_path)

    if args.no_cudnn_deterministic:
        print('WARNING: CUDNN Deterministic will be disabled.')
        utils.cudnn_deterministic = False

    if args.approach == 'lwm':
        assert args.network not in netmodels, 'LWM is not supported for TC'
    
    assert (args.network is None) ^ (args.ml_model is None), '--network and --ml-model are mutually exclusive.'  # ^ is xor
    
    if args.ml_model is not None:
        assert args.approach in ['scratch', 'multiclass_classifiers'], 'ML models are only supported by Scratch and Multiclass-Classifiers approaches.'
        print('WARNING: decomment assert on --statistical')
#         assert args.statistical, 'ML models accept only statistical input (pass --statistical).'

    utils.seed_everything(seed=args.seed)
    print('=' * 108)
    print('Arguments =')
    for arg in np.sort(list(vars(args).keys())):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 108)
    
    print('Extra Arguments =')
    print(extra_args)
    print('=' * 108)

    # Args -- CUDA
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = 'cuda'
    else:
        print('WARNING: [CUDA unavailable] Using CPU instead!')
        device = 'cpu'
    # Multiple gpus
    # if torch.cuda.device_count() > 1:
    #     self.C = torch.nn.DataParallel(C)
    #     self.C.to(self.device)
    ####################################################################################################################

    base_model_path = None
    if args.load_base_model:
        assert args.nc_first_task > 0, '--load-base-model is available only if --nc-first-task has been set.'

        from glob import glob
        import json
        import functools

        print('WARNING: optimizer, early stopping, and feature vector size params are not considered retrieving'
              'pretrained base models.')

        def check_train_options(curr_args, target_args, nc_first_task=True):
            train_opts = (
                    ['seed', 'no_cudnn_deterministic', 'datasets', 'batch_size', 'num_bytes', 'num_pkts', 'fields',
                     'network', 'keep_existing_head', 'pretrained', 'pre_allocated_output', 'last_class'] + (
                        ['nc_first_task'] if nc_first_task else []))
            return functools.reduce(lambda x, y: x & y, [getattr(curr_args, k) == target_args[k] for k in train_opts])

        print('INFO: loading of the pretrained base model will be tried')
        # Retrieving all args for pre-trained base models
        _base_model_path = args.base_model_path or os.path.join('%s_UB' % args.results_path.rstrip('/'))
        args_dict_fns = glob(os.path.join(_base_model_path, '**', 'args-*.txt'), recursive=True)
        if not len(args_dict_fns):
            print('WARNING: be sure the base models are stored in %s_UB. Continuing from the scratch.'
                  % args.results_path.rstrip('/'))
            args.load_base_model = False
        else:
            args_dict = None
            for args_dict_fn in args_dict_fns:
                with open(args_dict_fn) as fin:
                    _args_dict = json.loads(fin.read())
                if check_train_options(args, _args_dict, nc_first_task=True):
                    args_dict = _args_dict
                    break
            if args_dict is None:
                for args_dict_fn in args_dict_fns:
                    with open(args_dict_fn) as fin:
                        _args_dict = json.loads(fin.read())
                    if check_train_options(args, _args_dict, nc_first_task=False):
                        args_dict = _args_dict
                        break
            assert args_dict is not None, 'Be sure the base models are stored in %s' % _base_model_path
            nc_base_models = [
                args_dict['nc_first_task'] + t * args_dict['nc_incr_tasks'] for t in range(args_dict['num_tasks'])]
            assert args.nc_first_task in nc_base_models, 'Required base model is not available: %s' % nc_base_models
            target_base_task = nc_base_models.index(args.nc_first_task)
            ts = args_dict_fn.split('-')[-1].split('.')[0]
            base_model_fn = glob(os.path.join(_base_model_path, '**', 'task%s-%s*' % (target_base_task, ts)),
                                 recursive=True)
            assert len(base_model_fn) != 0, 'No base model matches the required.'
            assert len(base_model_fn) == 1, 'More than one base model matches the required.'
            base_model_path = base_model_fn[0]
            
    base_kwargs = dict(
        nepochs=args.nepochs, lr=args.lr, lr_min=args.lr_min, lr_factor=args.lr_factor,
        first_lr=args.first_lr, first_lr_min=args.first_lr_min, first_lr_factor=args.first_lr_factor,
        lr_patience=args.lr_patience, clipgrad=args.clipping, momentum=args.momentum,
        first_momentum=args.first_momentum, wd=args.weight_decay, multi_softmax=args.multi_softmax,
        wu_nepochs=args.warmup_nepochs, wu_lr_factor=args.warmup_lr_factor, fix_bn=args.fix_bn,
        eval_on_train=args.eval_on_train, predict_tasks=args.predict_tasks, statistical=args.statistical, is_ml=is_ml)

    # Args -- Store current commit hash
    try:
        repo = git.Repo(search_parent_directories=True)
        repo_sha1 = repo.head.commit.hexsha
        setattr(args, 'git_sha1', repo_sha1)
    except:
        setattr(args, 'git_sha1', 'NO_REPO')

    # Args -- Network
    if args.network is not None:
        if args.approach not in approach.model_growth_apprs:
            from networks.network import LLL_Net as Model
        else:
            from networks.multi_network import LLL_MultiNet as Model
    else:
        if args.approach not in approach.model_growth_apprs:
            from ml_models.ml_model import ML_Model as Model
        else:
            from ml_models.multi_ml_model import ML_MultiModel as Model

    mlm_args = argparse.Namespace()
    if args.network in tvmodels:  # torchvision models
        tvnet = getattr(importlib.import_module(name='torchvision.models'), args.network)
        if args.network == 'googlenet':
            init_model = tvnet(pretrained=args.pretrained, aux_logits=False)
        else:
            init_model = tvnet(pretrained=args.pretrained)
        set_tvmodel_head_var(init_model)
    elif args.network in netmodels:
        net = getattr(importlib.import_module(name='networks'), args.network)
        # WARNING: fixed to pretrained False for other model (non-torchvision)
        init_model = net(pretrained=False, num_bytes=args.num_bytes, num_pkts=args.num_pkts,
                         num_fields=len(args.fields), out_features_size=args.out_features_size, scale=args.scale)
    elif args.ml_model is not None:
        mlm = getattr(importlib.import_module(name='ml_models'), args.ml_model)
        mlm_args, extra_args = mlm.extra_parser(extra_args)
        init_model = mlm(**mlm_args.__dict__)
    else:  # other models declared in networks package's init
        net = getattr(importlib.import_module(name='networks'), args.network)
        # WARNING: fixed to pretrained False for other model (non-torchvision)
        init_model = net(pretrained=False)
        
    # Args -- Continual Learning Approach
    from approach.incremental_learning import Inc_Learning_Appr
    Appr = getattr(importlib.import_module(name='approach.' + args.approach), 'Appr')
    assert issubclass(Appr, Inc_Learning_Appr)
    appr_args, extra_args = Appr.extra_parser(extra_args)
    print('Approach arguments =')
    for arg in np.sort(list(vars(appr_args).keys())):
        print('\t' + arg + ':', getattr(appr_args, arg))
    print('=' * 108)

    # Args -- Exemplars Management
    from datasets.exemplars_dataset import ExemplarsDataset
    Appr_ExemplarsDataset = Appr.exemplars_dataset_class()
    if Appr_ExemplarsDataset:
        assert issubclass(Appr_ExemplarsDataset, ExemplarsDataset)
        appr_exemplars_dataset_args, extra_args = Appr_ExemplarsDataset.extra_parser(extra_args)
        print('Exemplars dataset arguments =')
        for arg in np.sort(list(vars(appr_exemplars_dataset_args).keys())):
            print('\t' + arg + ':', getattr(appr_exemplars_dataset_args, arg))
        print('=' * 108)
    else:
        appr_exemplars_dataset_args = argparse.Namespace()

    if len([v for v in args.datasets if v in ['mirage_generic']]) > 0:
        setattr(appr_exemplars_dataset_args, 'is_networking', True)

    # Args -- GridSearch
    if args.gridsearch_tasks > 0:
        from gridsearch import GridSearch
        gs_args, extra_args = GridSearch.extra_parser(extra_args)
        Appr_finetuning = getattr(importlib.import_module(name='approach.finetuning'), 'Appr')
        assert issubclass(Appr_finetuning, Inc_Learning_Appr)
        GridSearch_ExemplarsDataset = Appr.exemplars_dataset_class()
        print('GridSearch arguments =')
        for arg in np.sort(list(vars(gs_args).keys())):
            print('\t' + arg + ':', getattr(gs_args, arg))
        print('=' * 108)

    assert len(extra_args) == 0, 'Unused args: {}'.format(' '.join(extra_args))
    ####################################################################################################################

    # Log all arguments
    full_exp_name = reduce((lambda x, y: x[0] + y[0]), args.datasets) if len(args.datasets) > 0 else args.datasets[0]
    full_exp_name += '_' + args.approach
    if args.exp_name is not None:
        full_exp_name += '_' + args.exp_name
    logger = MultiLogger(args.results_path, full_exp_name, loggers=args.log, save_models=args.save_models)
    logger.log_args(argparse.Namespace(
        **args.__dict__,
        **appr_args.__dict__,
        **appr_exemplars_dataset_args.__dict__,
        **mlm_args.__dict__))

    # Loaders
    utils.seed_everything(seed=args.seed)
    trn_loader, val_loader, tst_loader, taskcla = get_loaders(
        args.datasets, args.num_tasks, args.nc_first_task, args.nc_incr_tasks, args.batch_size,
        validation=args.validation, num_workers=args.num_workers, pin_memory=args.pin_memory, num_bytes=args.num_bytes,
        num_pkts=args.num_pkts, fields=args.fields, seed=args.seed, shots=args.shots, fseed=args.fseed,
        last_class=args.last_class, predict_tasks=args.predict_tasks, statistical=args.statistical, is_ml=is_ml
        # , apply_scaling=args.network != 'Chen21RNN'
    )
        
    # Apply arguments for loaders
    if args.use_valid_only:
        tst_loader = val_loader
    max_task = len(taskcla) if args.stop_at_task == 0 else args.stop_at_task

    # TODO: solve the activate_features args within the single approach file, not here.
    # Network and Approach instances
    utils.seed_everything(seed=args.seed)
    model = Model(init_model, remove_existing_head=not args.keep_existing_head,
                  activate_features=args.approach != 'lucir')
    try:
        modspec_models = [
            Model(init_model.modspec_models[m], remove_existing_head=not args.keep_existing_head, modality=m,
                    activate_features=args.approach != 'lucir')
            for m in range(len(init_model.modspec_models))]
    except AttributeError as _:
        modspec_models = None
    utils.seed_everything(seed=args.seed)
    # taking transformations and class indices from first train dataset
    first_train_ds = trn_loader[0].dataset
    transform, class_indices = first_train_ds.transform, first_train_ds.class_indices
    appr_kwargs = {**base_kwargs, **dict(logger=logger, **appr_args.__dict__)}
    if Appr_ExemplarsDataset:
        appr_kwargs['exemplars_dataset'] = Appr_ExemplarsDataset(transform, class_indices,
                                                                 **appr_exemplars_dataset_args.__dict__)
    if args.approach == 'lucir' and args.nc_incr_tasks == 1:
        print('WARNING: when performing 1-class increments, the K parameter for LUCIR is forced to 1.')
        appr_kwargs['K'] = 1

    utils.seed_everything(seed=args.seed)
    # Limiting the number of used cpu
    if device == 'cpu':
        torch.set_num_threads(args.num_workers)

    appr = Appr(model, device, **appr_kwargs)
    try:
        appr_kwargs['nepochs'] = args.modspec_nepochs
        modspec_apprs = [Appr(modspec_models[m], device, **appr_kwargs) for m in range(len(modspec_models))]
    except TypeError as _:
        modspec_apprs = None

    # GridSearch
    if args.gridsearch_tasks > 0:
        ft_kwargs = {**base_kwargs, **dict(logger=logger,
                                           exemplars_dataset=GridSearch_ExemplarsDataset(transform, class_indices))}
        appr_ft = Appr_finetuning(model, device, **ft_kwargs)
        gridsearch = GridSearch(appr_ft, args.seed, gs_args.gridsearch_config, gs_args.gridsearch_acc_drop_thr,
                                gs_args.gridsearch_hparam_decay, gs_args.gridsearch_max_num_searches)

    # Loop tasks
    acc_taw = np.zeros((max_task, max_task))
    acc_tag = np.zeros((max_task, max_task))
    forg_taw = np.zeros((max_task, max_task))
    forg_tag = np.zeros((max_task, max_task))
    for t, (_, ncla) in enumerate(taskcla):
        # Early stop tasks if flag
        if t >= max_task:
            continue

        print('*' * 108)
        print('Task {:2d}'.format(t))
        print('*' * 108)

        if not args.pre_allocated_output:
            # Add head for current task
            model.add_head(taskcla[t][1], binary=args.approach in ova_mg_approaches)
            if modspec_models:
                for modspec_model in modspec_models:
                    modspec_model.add_head(taskcla[t][1], binary=args.approach in ova_mg_approaches)
        elif t == 0:
            # Add a single pre-allocated head, which is composed of multiple heads
            for tc in taskcla[:max_task]:
                model.add_head(tc[1], binary=args.approach in ova_mg_approaches)
                if modspec_models:
                    for modspec_model in modspec_models:
                        modspec_model.add_head(tc[1], binary=args.approach in ova_mg_approaches)
        model.to(device)
        if modspec_models:
            for modspec_model in modspec_models:
                modspec_model.to(device)

        def apply_gridsearch(gridsearch, appr, t, trn_loader, val_loader):
            # Search for best finetuning learning rate -- Maximal Plasticity Search
            print('LR GridSearch')
            best_ft_acc, best_ft_lr = gridsearch.search_lr(appr.model, t, trn_loader[t], val_loader[t])
            # Apply to approach
            appr.lr = best_ft_lr
            gen_params = gridsearch.gs_config.get_params('general')
            for k, v in gen_params.items():
                if not isinstance(v, list):
                    setattr(appr, k, v)

            # Search for best forgetting/intransigence tradeoff -- Stability Decay
            print('Trade-off GridSearch')
            best_tradeoff, tradeoff_name = gridsearch.search_tradeoff(args.approach, appr,
                                                                      t, trn_loader[t], val_loader[t], best_ft_acc)
            # Apply to approach
            if tradeoff_name is not None:
                setattr(appr, tradeoff_name, best_tradeoff)

            print('-' * 108)

        # Train
        if modspec_apprs and not (args.load_base_model and t == 0):
            # modspec_weights = []
            for m, modspec_appr in enumerate(modspec_apprs):
                print('_' * 108)
                print('Modality %d' % (m + 1))
                print('_' * 108)

                # Unfreeze all the modality layers: each approach will manage freezing internally
                modspec_appr.model.unfreeze_all(t)
                # GridSearch
                if t < args.gridsearch_tasks:
                    apply_gridsearch(gridsearch, modspec_appr, t, trn_loader, val_loader)
                # Train the current modality model
                modspec_appr.train(t, trn_loader[t], val_loader[t])
                # Freeze all the modality layers but not the last, in order to proper fine-tune the remaining weights
                modspec_appr.model.freeze_all(t, non_freezing=['fc1'], verbose=False)

                try:
                    appr.model.model[t].model.modspec_models[m] = modspec_appr.model.model[t].model
                except TypeError as _:
                    pass

            appr.refresh_initial_weights()
            print('_' * 108)
            print('Shared modality')
            print('_' * 108)
        # GridSearch
        if t < args.gridsearch_tasks:
            apply_gridsearch(gridsearch, appr, t, trn_loader, val_loader)

        if not is_ml:
            appr.model.trainability_info()
        
        if not args.debug:
            appr.train(t, trn_loader[t], val_loader[t], base_model_path)
        else:
            appr.train(t, trn_loader[t], val_loader[t])

        print('-' * 108)

        # Test
        out_list = []
        tar_list = []
        features_list = []
        for u in range(t + 1):
            
            evalclock1 = time.process_time()
            test_loss, acc_taw[t, u], acc_tag[t, u], outputs, targets, features = appr.eval(u, tst_loader[u])
            evalclock2 = time.process_time()
            out_list.append(outputs)
            tar_list.append(targets)
            features_list.append(features)

            if u < t:
                forg_taw[t, u] = acc_taw[:t, u].max(0) - acc_taw[t, u]
                forg_tag[t, u] = acc_tag[:t, u].max(0) - acc_tag[t, u]
            print('>>> Test on task {:2d} : loss={:.3f} | TAw acc={:5.3f}%, forg={:5.3f}%'
                  '| TAg acc={:5.3f}%, forg={:5.3f}%, time= {:5.3f}, Num_instances= {:2d} <<<'.format(
                u, test_loss, 100 * acc_taw[t, u], 100 * forg_taw[t, u], 100 * acc_tag[t, u], 100 * forg_tag[t, u],
                              evalclock2 - evalclock1, len(targets)))
            logger.log_scalar(task=t, iter=u, name='loss_' + str(args.seed), group='test', value=test_loss)
            logger.log_scalar(task=t, iter=u, name='acc_taw_' + str(args.seed), group='test', value=100 * acc_taw[t, u])
            logger.log_scalar(task=t, iter=u, name='acc_tag_' + str(args.seed), group='test', value=100 * acc_tag[t, u])
            logger.log_scalar(task=t, iter=u, name='forg_taw_' + str(args.seed), group='test',
                              value=100 * forg_taw[t, u])
            logger.log_scalar(task=t, iter=u, name='forg_tag_' + str(args.seed), group='test',
                              value=100 * forg_tag[t, u])

        # Save
        print('Save at ' + os.path.join(args.results_path, full_exp_name))
        logger.log_result(acc_taw, name='acc_taw_' + str(args.seed), step=t)
        logger.log_result(acc_tag, name='acc_tag_' + str(args.seed), step=t)
        logger.log_result(forg_taw, name='forg_taw_' + str(args.seed), step=t)
        logger.log_result(forg_tag, name='forg_tag_' + str(args.seed), step=t)
        try:
            logger.save_model(model.state_dict(), task=t)
        except:
            print('WARNING: model not saved.')
        logger.log_result(acc_taw.sum(1) / np.tril(np.ones(acc_taw.shape[0])).sum(1),
                          name='avg_accs_taw_' + str(args.seed), step=t)
        logger.log_result(acc_tag.sum(1) / np.tril(np.ones(acc_tag.shape[0])).sum(1),
                          name='avg_accs_tag_' + str(args.seed), step=t)
        aux = np.tril(np.repeat([[tdata[1] for tdata in taskcla[:max_task]]], max_task, axis=0))
        logger.log_result((acc_taw * aux).sum(1) / aux.sum(1), name='wavg_accs_taw_' + str(args.seed), step=t)
        logger.log_result((acc_tag * aux).sum(1) / aux.sum(1), name='wavg_accs_tag_' + str(args.seed), step=t)

        # Last layer analysis
        if args.last_layer_analysis:
            weights, biases = last_layer_analysis(model.heads, t, taskcla, y_lim=True)
            logger.log_figure(name='weights_' + str(args.seed), iter=t, figure=weights)
            logger.log_figure(name='bias_' + str(args.seed), iter=t, figure=biases)

            # Output sorted weights and biases
            weights, biases = last_layer_analysis(model.heads, t, taskcla, y_lim=True, sort_weights=True)
            logger.log_figure(name='weights_sorted_' + str(args.seed), iter=t, figure=weights)
            logger.log_figure(name='bias_sorted_' + str(args.seed), iter=t, figure=biases)

        # save scores, targets and features for each task
        df = pd.DataFrame({'Scores': [out_list], 'Targets': [tar_list], 'Features': [features_list]})
        logger.log_parquet(df, name='outputs_targets_features_' + str(args.seed), task=t)

    # Print Summary
    utils.print_summary(acc_taw, acc_tag, forg_taw, forg_tag)
    print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))
    print('Done!')

    return acc_taw, acc_tag, forg_taw, forg_tag, logger.exp_path
    ####################################################################################################################


if __name__ == '__main__':
    main()
