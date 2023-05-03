import functools
import json
from copy import deepcopy

import numpy as np
import pandas as pd
from tqdm import tqdm

appr_dict = {'icarl': 'iCaRL', 'icarlo': 'iCaRL-original', 'icarlp': 'iCaRL+', 'bic': 'BiC', 'il2m': 'IL2M',
             'lwf': 'LwF', 'finetuning': 'FT+', 'lucir': 'LUCIR', 'ewc': 'EWC', 'joint': 'Joint', 'lwfgkd': 'LwF-GKD',
             'scratch': 'Scratch', 'freezing': 'FZ+', 'eeil': 'EEIL', 'ssil': 'SSIL',
             'bicp': 'BiC-LUCIR', 'scratchmem': 'Scratch w/ Limitate Memory',
             'jointmem': 'FT-Mem', 'backbonefreezingmem': 'FZ-Mem',
             'scratchmembal': 'Scratch w/ Limitate Memory + Balance', 'jointmembal': 'FT-Mem + Balance',
             'bicpp': 'BiC-Tanh', 'jointft': 'FT', 'backbonefreezing': 'FZ',
             'finetuningp': 'FineTuning-SigBCE', 'freezingp': 'FeatsExtr-SigBCE', 'ssilp': 'SSIL-SigBCE',
             'lwfp': 'LwF-SigBCE', 'wu2022': 'Wu22', '_': '_', 'chen2021': 'OvA-Ens', 'OvA-Ens': 'Chen21',
             'multiclass_classifiers': 'MC', 'label_halluc': 'LabelHalluc', 'label_halluc_aug': 'AugmentLabelHalluc',
             'augment': 'SampleAugment'
            }

appr_dict_r = dict([(v, k) for k, v in appr_dict.items()])
appr_list_alph_sorted = sorted(appr_dict.items(), key=lambda x: x[1].lower(), reverse=False)
appr_dict_alph_sorted = {val[0]: val[1] for val in appr_list_alph_sorted}
metr_dict = {'accuracy_score': 'Accuracy', 'balanced_accuracy_score': 'Balanced Accuracy', 'f1_score': 'F1 Score',
             'precision_score': 'Precision', 'recall_score': 'Recall', 'top_k_accuracy_score': 'Top k Accuracy'}

metr_dict.update(
    dict([('forgetting_%s' % k, '%s Forgetting' % v) for k, v in metr_dict.items()])
)

metr_dict['davies_bouldin_score'] = 'Davies-Bouldin Index'

mety_dict = {
    'normal': {
        'all': '',
        'old': 'Old',
        'new': 'New',
        'base': 'Base',
        'per_episode_metrics': 'Per Episode',
        'per_class_metrics': 'Macro'
    },
    'drop': {
        'all': 'Drop',
        'old': 'Drop Old',
        'new': 'Intransigence',
        'base': 'Drop Base',
        'per_episode_metrics': 'Per Episode Intransigence',
        'per_class_metrics': 'Macro Intransigence'
    }
}

cols_dict = {'approach': 'Approach', 'network': 'Model', 'ml_model': 'Model', 'nc_first_task': 'Base',
             'nc_incr_tasks': 'Increment', 'batch_size': 'Batch Size', 'lr_patience': 'Patience', 'type': 'Type',
             'out_features_size': 'Out Features Size', 'momentum': 'Momentum', 'lamb': 'Base Lambda'}


def get_approaches(analysis=None):
    if analysis == 'naive':
        # Naive approaches analysis
        return ['finetuning', 'scratch', 'freezing', 'jointft', 'backbonefreezing', 'joint']
    if analysis == 'ranking':
        # Ranking
        return ['icarl', 'icarlp', 'bic', 'il2m', 'lwf', 'lucir', 'ewc', 'scratch', 'eeil', 'ssil',
                'chen2021', 'finetuning', 'freezing', 'jointft', 'backbonefreezing', 'joint']
    if analysis == 'lineplot':
        # Lineplot
        return ['icarl', 'bic', 'il2m', 'lwf', 'lucir', 'ewc', 'scratch', 'eeil', 'ssil']
    if analysis == 'lineplot_net':
        # Lineplot Networking
        return ['wu2022', 'chen2021', 'icarlp', 'scratch']
    return ['icarl', 'icarlp', 'bic', 'il2m', 'lwf', 'lucir', 'ewc', 'scratch', 'eeil', 'ssil', 'finetuning',
            'freezing', 'jointft', 'backbonefreezing', 'wu2022', 'chen2021', 'joint']


def get_ts(df_filename):
    return '-'.join(df_filename.split('/')[-1].split('-')[1:]).split('_')[0].split('.')[0]


def get_args_dict(df_filename):
    ts = get_ts(df_filename)
    try:
        with open('/'.join(df_filename.split('/')[:-2] + ['args-%s.txt' % ts])) as fin:
            return json.loads(fin.read())
    except FileNotFoundError as _:
        with open('/'.join(df_filename.split('/')[:-1] + ['args-%s.txt' % ts])) as fin:
            return json.loads(fin.read())


def get_stdout_list(df_filename):
    ts = get_ts(df_filename)
    try:
        with open('/'.join(df_filename.split('/')[:-2] + ['stdout-%s.txt' % ts])) as fin:
            return [v.strip() for v in fin.readlines()]
    except FileNotFoundError as _:
        with open('/'.join(df_filename.split('/')[:-1] + ['stdout-%s.txt' % ts])) as fin:
            return [v.strip() for v in fin.readlines()]


def discard_duplicates(filenames, take_newest=True):
    filenames_dict = dict()
    for filename in filenames:
        try:
            args_dict = get_args_dict(filename)
        except:
            continue
        key = (
            '%s' % args_dict['approach'],
            '%s' % args_dict['network'],
            '%s' % args_dict.get('ml_model', 'none'),
            '%d' % args_dict['seed'],
            '%d' % args_dict.get('fseed', -1),
            '%d' % args_dict.get('shots', -1),
            '%d' % args_dict.get('last_class', -1),
            '%d' % args_dict['nc_first_task'],
            '%d' % args_dict['nc_incr_tasks'],
            '%d' % args_dict['num_exemplars'],
            # '%s' % args_dict.get('knowledge_distillation', 'none'),
        )
        if args_dict['approach'] == 'label_halluc':
            key += (
                '%d' % args_dict['num_ft_epochs'],
                '%.4f' % args_dict['ft_lr'],
                '%.4f' % args_dict['ft_lr_min'],
                '%.2f' % args_dict['ft_lr_factor'],
                '%d' % args_dict['ft_lr_patience'],
                '%s' % args_dict['augment'],
                '%s' % args_dict['augment_type'],
                '%d' % args_dict['augment_factor'],
                '%d' % args_dict['augment_after'],
                '%s' % args_dict['no_halluc'],
                '%d' % args_dict['halluc_after'],
            )
        if key not in filenames_dict:
            filenames_dict[key] = filename
        else:
            curr_ts = int(filename.split('/')[-1].split('.')[0].split('-')[-1].split('_')[0])
            old_ts = int(filenames_dict[key].split('/')[-1].split('.')[0].split('-')[-1].split('_')[0])
            if (take_newest and curr_ts > old_ts) or (not take_newest and curr_ts < old_ts):
                filenames_dict[key] = filename
    _filenames = []
    # c = 0
    for key in filenames_dict:
        _filenames.append(filenames_dict[key])
        # if 'eeil' == key[0]:
           # print(filenames_dict[key], key)
           # c += 1
    # print(c)
    # input()
    # _filenames = []
    # for key in filenames_dict:
    #     _filenames.append('-'.join([key[0], filenames_dict[key], key[-1]]))
    # with open('try.txt', 'w') as fout:
    #     fout.writelines([v + '\n' for v in _filenames])
    # exit()
    return _filenames


def get_training_info(stdout_filenames):
    train_columns = ['Seed', 'Task', 'Epoch', 'Train Time', 'Validation Time', 'Validation Loss', 'TAw Accuracy',
                     'Learning Rate']
    timing_columns = ['Approach', 'Seed', 'Type', 'Task', 'Time', 'Num Samples']
    df_train = pd.DataFrame(columns=train_columns)
    df_timing = pd.DataFrame(columns=timing_columns)
    row_train_list = []
    row_timing_list = []
    for filename in tqdm(stdout_filenames):

        stdout_list = get_stdout_list(filename)
        
        if 'done' not in stdout_list[-1].lower():
            print('stdout file of %s is broken' % filename)
            continue
        
        args_dict = get_args_dict(filename)

        row_train = {'Approach': (args_dict['approach']),
                     'Seed': int(args_dict['seed']),
                     'Last App': int(args_dict.get('last_class', -1)),
                     'First Momentum': args_dict.get('first_momentum', 0),
                     'Base Momentum': args_dict.get('momentum', 0),
                     'Base Apps': args_dict['nc_first_task'],
                     'Increment': args_dict['nc_incr_tasks'],
                     'Memory Size': args_dict['num_exemplars']}
        row_timing = deepcopy(row_train)

        out_features_size = args_dict.get('out_features_size', -1)
        if out_features_size is None:
            out_features_size = -1
        row_train.update({'Model': args_dict['network'] or args_dict['ml_model'],
                          'Out Features Size': '%d' % out_features_size})
        row_timing.update({'Model': args_dict['network'] or args_dict['ml_model'],
                           'Out Features Size': '%d' % out_features_size})

        lr = float(args_dict['lr'])
        task = model_class = retrained_class = -1
        for line in stdout_list:
            try:
                line = line.replace(' ', '')
                if 'Task' in line and 'BiC' not in line:
                    task = int(line.lstrip('Task'))
                if 'Trainone-vs-rest' in line:
                    model_class = line.split('class')[1]
                if 'Retraining' in line:
                    retrained_class = line.split('class')[1]
                if 'Epoch' in line and 'Valid' in line:
                    line = line.strip('|').split('|')
                    row_train.update(
                        {
                            'Learning Rate': lr,
                            'Task': task,
                            'Epoch': int(line[0].split(',')[0].lstrip('Epoch')),
                            'Train Time': float(line[0].split(',')[1].split('=')[1].rstrip('s')),
                            'Validation Time': float(line[2].split('time=')[1].split('s')[0]),
                            'Validation Loss': float(line[2].split('loss=')[1].split(',')[0]),
                            'TAw Accuracy': float(line[2].split(',')[1].split('=')[1].rstrip('%')),
                            'Model App': model_class,
                            'Retrained App': retrained_class
                        })
                    row_train_list.append(deepcopy(row_train))
                    if 'lr' in line:
                        lr = float(line[3].split('=')[-1].replace(',', '.'))
                if 'TrainTime' in line:
                    line = line.strip('|').split(':')[1].split(',')
                    for i in np.arange(3):
                        row_timing.update(
                            {
                                'Task': task,
                                'Type': line[i].split('=')[0],
                                'Time': float(line[i].split('=')[1].rstrip('s'))
                            })
                        row_timing_list.append(deepcopy(row_timing))
                if 'Test' in line:
                    line = line.strip('<<<').split('|')[-1].split(',')
                    row_timing.update(
                        {
                            'Num Samples': int(line[3].split('=')[-1]),
                            'Task': task,
                            'Type': 'test',
                            'Time': float(line[2].split('=')[-1])
                        })
                    row_timing_list.append(deepcopy(row_timing))
            except:
                print(line)

    df_train = df_train.append(row_train_list, ignore_index=True)
    df_timing = df_timing.append(row_timing_list, ignore_index=True)
    return df_train, df_timing


def compute_means(values, base, incr, ep, t='All', per_episode=False):
    """
    values: list of per-class (per-episode) metrics
    base: number of base applications
    incr: size of increments
    ep: current episode, it is 0 for the base training
    t: type of required mean, it could be all (viz. macro average on the entire set of classes), old (viz. macro average
      on the set of old classes), new (viz. macro average on the set of new classes), or base (viz. macro average on
      on the set of base classes).
    per_episode: when true, values is treated episodically, instead (default) values should contain per-class metrics
    """
    # The first_class index is always zero
    first_class = 0
    # The first_new_class index is base at the episodes 0 and 1, base + (ep-1) * incr otherwise. When is per_episode the
    # value assigned is 0 for episodes 0 and 1, otherwise is ep
    first_new_class = (base + max([(ep - 1) * incr, 0])) if not per_episode else (max([ep - 1, 0]) + 1)
    # The last_class index is the next to the last class index, namely the total number of valid values
    last_class = (base + ep * incr) if not per_episode else ep + 1
    # The last_base_class index is the index of the episode 1 first value
    last_base_class = base if not per_episode else 1

    if t == 'All':
        start_idx, end_idx = first_class, last_class
    elif t == 'Old':
        start_idx, end_idx = first_class, first_new_class
    elif t == 'Base':
        start_idx, end_idx = first_class, last_base_class
    else:  # if t == 'New':
        start_idx, end_idx = first_new_class, last_class

    # last_* indexes are build in order to include the last element
    return np.nanmean(values[start_idx:end_idx]) if len(values[start_idx:end_idx]) else np.nan


def load_data(df_filenames, metric='f1_score', return_scenarios=False, per_episode=False, is_gate=False):
        
    df_filenames = discard_duplicates(df_filenames)
    
    rows = []
    for df_filename in tqdm(df_filenames):
        args_dict = get_args_dict(df_filename)
        tmp_df = pd.read_parquet(df_filename)

        if 'nc_first_task' in tmp_df and 'nc_incr_tasks' in tmp_df:
            tmp_df_g = tmp_df.groupby(['nc_first_task', 'nc_incr_tasks'])
        else:
            # If the columns 'nc_first_task' and 'nc_incr_tasks' are not present, the current df is associated w/ values
            # stored in the associated args file
            tmp_df_g = [((args_dict['nc_first_task'], args_dict['nc_incr_tasks']), tmp_df)]

        memory_size = args_dict.get('num_exemplars', 0)
        # Scratch w/ bounded memory is associated with scratchmem approach
        approach = args_dict['approach']\
            if args_dict['approach'] not in ['scratch', 'joint', 'backbonefreezing'] or memory_size == 0\
            else args_dict['approach'] + 'mem'
        if args_dict['approach'] == 'label_halluc':
            if args_dict.get('augment', False):
                if args_dict.get('no_halluc', False):
                    approach = 'augment'
                else:
                    approach = args_dict['approach'] + '_aug'
        balance_new_tasks = args_dict.get('balance_new_tasks', False)
        approach = approach if not balance_new_tasks else approach + 'bal'

        knowledge_distillation = args_dict.get('knowledge_distillation', 'none')
        approach = approach if approach != 'lwf' or knowledge_distillation in ['tkd', 'none'] else 'lwfgkd'

        row = {
            'Seed': args_dict['seed'],
            'Model':  args_dict['network'] or args_dict['ml_model'],
            'Approach': '%s' % appr_dict[approach],
            'Batch Size': args_dict['batch_size'],
            'Patience': args_dict['lr_patience'],
            'Out Features Size': args_dict.get('out_features_size', 200),
            'Momentum': args_dict.get('momentum', .0),
            'Base Lambda': args_dict.get('lamb', -1),
            'Last App': args_dict.get('last_class', -1),
            'Memory Size': memory_size,
            'Balance New Tasks': balance_new_tasks,
            'fSeed': args_dict.get('fseed', -1),
            'Shots': args_dict.get('shots', -1),
            'AugmentType': args_dict.get('augment_type', None),
            'AugmentFact': args_dict.get('augment_factor', None),
        }

        for g, tmp_d in tmp_df_g:
            _nc_first_task = g[0]
            _nc_incr_tasks = g[1]

            row.update({'Base Apps': _nc_first_task, 'Increment': _nc_incr_tasks})
            
            if is_gate:
                _nc_first_task = _nc_incr_tasks = 1

            values = [[(v0 if v0 is not None else np.nan) * 100 for v0 in v] for v in tmp_d[metric].values]
            for ep, value in enumerate(values):
                for t in ['Base', 'Old', 'New', 'All']:
                    row.update({
                        'Episode': ep,
                        # The current number of apps is the base classes + the eventual increments, ep is 0 for the base
                        '#Apps': _nc_first_task + _nc_incr_tasks * ep,
                        metr_dict[metric]: compute_means(value, _nc_first_task, _nc_incr_tasks, ep, t, per_episode),
                        'Type': t
                    })

                    rows.append(deepcopy(row))
    df = pd.DataFrame(rows)

    if return_scenarios:
        # The scenarios is identified by the number of base applications and by the size of increments
        # Returning scenarios is useful to properly build upperbound metrics
        return df, list(df.groupby(['Base Apps', 'Increment']).groups)
    else:
        return df


def load_scratch_data(df_filenames, metric='f1_score', base=20, incr=5, max_classes=40):
    df_filenames = discard_duplicates(df_filenames)

    # Stop is computed basing on the base classes and the increment size up to reach the max_classes
    stop = 1 + (((max_classes - base) // incr) if incr > 0 else 0)

    df = pd.DataFrame()
    df_tmp = pd.DataFrame()
    rows = []
    for df_filename in tqdm(df_filenames):
        args_dict = get_args_dict(df_filename)
        tmp_df = pd.read_parquet(df_filename)

        if 'nc_first_task' in tmp_df:
            tmp_df_g = tmp_df.groupby(['nc_first_task', 'nc_incr_tasks'])
        else:
            tmp_df_g = [((args_dict['nc_first_task'], args_dict['nc_incr_tasks']), tmp_df)]

        line = None
        approach = 'scratch'
        # Retrieving the class order list from the current stdout file
        for line in get_stdout_list(df_filename):
            if 'class order' in line.lower():
                break
        class_order = eval(line.split(':')[-1])

        row = {'Seed': args_dict['seed'],
               'Last App': args_dict.get('last_class', -1),
               'Model':  args_dict['network'] or args_dict['ml_model'],
               'Approach': '%s' % appr_dict[approach],
               'Batch Size': args_dict['batch_size'],
               'Patience': args_dict['lr_patience'],
               'Out Features Size': args_dict.get('out_features_size', 200),
               'Momentum': args_dict.get('momentum', .0),
               'Memory Size': args_dict.get('num_exemplars', 0),
               'App Order': class_order,
               }

        for g, tmp_d in tmp_df_g:
            _nc_first_task = g[0]
            _nc_incr_tasks = g[1]

            row.update({'Base Apps': _nc_first_task, 'Increment': _nc_incr_tasks})

            values = [[v0 * 100 for v0 in v] for v in tmp_d[metric].values]

            for ep, value in enumerate(values):
                # For each episode we store only number of apps and the list of per-class (per-episode) metrics
                row.update({
                    '#Apps': _nc_first_task + _nc_incr_tasks * ep,
                    metr_dict[metric]: value,
                })
                rows.append(deepcopy(row))
    df_tmp = df_tmp.append(rows, ignore_index=True)

    assert len(df_tmp) > 0

    # Basing on the passed arguments, the cumulative number of classes represents the number of classes at each episode
    cumnum_classes = list(range(base, base + incr * (stop - 1) + 1, incr)) if incr > 0 else [base]
    print(cumnum_classes)
    df_tmp = df_tmp[df_tmp['#Apps'].isin(cumnum_classes)]

    rows = []
    # This loop is needed in order to build episodes for the scratch approach
    for ep, nc in enumerate(cumnum_classes):
        # The tmp_df is grouped basing on several columns in order to separate configurations
        group_columns = ['Model', 'Out Features Size', 'Memory Size', '#Apps', 'Approach', 'Batch Size',
                         'Patience', 'Momentum']
        # The scenario 39+1 needs an alignment of per-class (per-episode) metrics
        if base == 39 and incr == 1 and stop == 2:
            # The Last App column is needed to separate different executions
            group_columns = ['Last App'] + group_columns
            df_g = df_tmp[df_tmp['#Apps'] == nc].groupby(group_columns)

            for g in df_g.groups:
                # Scratch for 39 classes and no last class is discarded, because for the base model we need the one w/
                # the correct Last App
                if g[group_columns.index('Last App')] == -1 and nc == 39:
                    continue

                # # For the 40 classes we process the sole seed 0
                # if g[group_columns.index('Seed')] != 0 and nc == 40:
                #     continue

                # assert len(df_g.groups[g]) == 1, "Maybe some duplicate is present..."

                # values = df_tmp.loc[df_g.groups[g], metr_dict[metric]].values[0]
                # class_orders = df_tmp.loc[df_g.groups[g], 'App Order'].values[0]
                #
                # # Sorting values by class_order
                # values = [values[j] for j in np.argsort(class_orders[:nc])]

                values = df_tmp.loc[df_g.groups[g], metr_dict[metric]].values
                class_orders = df_tmp.loc[df_g.groups[g], 'App Order'].values

                for i in range(len(df_g.groups[g])):
                    values[i] = np.array([values[i][j] for j in np.argsort(class_orders[i][:nc])])

                values = np.mean(values, axis=0).tolist()

                # When episode is 1 (namely 1 increment from 39 base classes), each class is selected as last
                # else the old behavior is maintained for the 39 base classes
                for l in range(40) if ep == 1 else [40]:
                    # For each last_class the values vector is arranged in order to put the last_class at the end
                    _values = (values[:l] + values[l + 1:] + values[l:l + 1])[:nc]
                    for t in ['Base', 'All', 'Old', 'New']:
                        if t == 'All':
                            value = np.mean(_values)
                        if t == 'New':
                            value = np.mean(_values[39:])
                        else:  # t in ['Old', 'Base']:
                            value = np.mean(_values[:39])

                        row = dict([(k, v) for k, v in zip(group_columns, g)])
                        row.update({
                            'Seed': 0,
                            'Base Apps': 39,
                            'Increment': 1,
                            'Episode': ep,
                            'Type': t,
                            metr_dict[metric]: value})
                        if l < 40:
                            # The Last App is updated only when building the 40 iterations from the model @40
                            row.update({'Last App': l})
                        rows.append(deepcopy(row))
        else:
            group_columns = ['Seed'] + group_columns
            df_g = df_tmp[df_tmp['#Apps'] == nc].groupby(group_columns)
            for g in df_g.groups:
                for t in ['Base', 'All', 'Old', 'New']:
                    values = df_tmp.loc[df_g.groups[g], metr_dict[metric]].values
                    values = np.mean(values, axis=0)

                    row = dict([(k, v) for k, v in zip(group_columns, g)])
                    row.update({'Last App': -1,
                                'Base Apps': base,
                                'Increment': incr,
                                'Episode': ep,
                                'Type': t,
                                metr_dict[metric]: compute_means(values, base, incr, ep, t)})
                    rows.append(deepcopy(row))

    df = df.append(rows, ignore_index=True)

    return df


def compute_drops(df, df_UB, metric='f1_score'):
    columns = ['Seed', 'Last App', 'Model', 'Type', '#Apps']
    df_g = df.groupby(columns)
    groups = list(df_g.groups)
    del df_g
    for group in groups:
        df_filter = functools.reduce(lambda x, y: x & y, [df[col] == g for col, g in zip(columns, group)])
        df_UB_filter = functools.reduce(lambda x, y: x & y, [df_UB[col] == g for col, g in zip(columns, group)])
        if not len(df_UB.loc[df_UB_filter]):
            print('WARNING: group (%s) does not present Upperbound model.' % ', '.join([str(v) for v in group]))
            continue
        else:
            ub_value = df_UB.loc[df_UB_filter, metr_dict[metric]].values[0]
            df.loc[df_filter, '%s Drop' % metr_dict[metric]] = df.loc[
                df_filter, metr_dict[metric]].apply(lambda x: ub_value - x)
    return df
