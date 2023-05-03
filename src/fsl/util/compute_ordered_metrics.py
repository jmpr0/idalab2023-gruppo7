import os
import sys
import pandas as pd
from datasets import dataset_config
import glob
import json
import numpy as np


def get_class_order_from_log(log_path, dconf, dset='mirage_generic'):
    class_order_set = []
    with open(log_path, 'r') as fin:

        for line in fin:
            if 'Class order' in line:
                class_order = line.replace('Class order:  [', '').replace(']', '').split(',')
                class_order = [int(c) for c in class_order]
                for c in class_order:
                    if c in dconf[dset]['fs_split']['train_classes']:
                        class_order_set.append(0)
                    elif c in dconf[dset]['fs_split']['val_classes']:
                        class_order_set.append(1)
                    elif c in dconf[dset]['fs_split']['test_classes']:
                        class_order_set.append(2)
                    else:
                        print('Something goes wrong')
                        break
                return class_order, class_order_set
            else:
                continue


def get_ordered_macro_metrics(res_path: str, seeds: list):
    df_filename = glob.glob(os.path.join(res_path, 'results', 'outputs_targets_*_per_class_metrics.parquet'))[0]
    metrics_list = [met for met in pd.read_parquet(df_filename).columns.tolist() if met != 'confusion_matrix']
    metrics_per_seed_dict = {met + '_per_seed': None for met in metrics_list}
    for key in metrics_per_seed_dict:
        metrics_per_seed_dict[key] = {'train': [], 'validation': [], 'test': []}
    set_type_dict = {0: 'train', 1: 'validation', 2: 'test'}
    args_filenames = glob.glob(os.path.join(res_path, 'args-*.txt'))
    for args_filename in args_filenames:
        with open(args_filename, 'r') as json_file:
            args_dict = json.load(json_file)
        if args_dict['seed'] in seeds:
            ts = os.path.basename(os.path.splitext(args_filename)[0]).split('-')[1]
            stdout_filename = os.path.join(res_path, 'stdout-%s.txt' % ts)
            cord, cord_set = get_class_order_from_log(stdout_filename, dataset_config.dataset_config)
            df_path = os.path.join(res_path, 'results',
                                   'outputs_targets_features%s-%s_per_class_metrics.parquet' % (args_dict['seed'], ts))
            df = pd.read_parquet(df_path)
            for metric in metrics_list:
                if metric in df.columns:
                    last_metric_vals = df.iloc[-1][metric]
                    single_seed_met_set = {set_type_dict[set_type]: [] for set_type in sorted(set(cord_set))}
                    for set_type_num, met_val in zip(cord_set, last_metric_vals):
                        single_seed_met_set[set_type_dict[set_type_num]].append(met_val)
                    for set_type in metrics_per_seed_dict[metric + '_per_seed']:
                        metrics_per_seed_dict[metric + '_per_seed'][set_type]. \
                            append(np.mean(single_seed_met_set[set_type]))
                else:
                    for set_type in metrics_per_seed_dict[metric + '_per_seed']:
                        metrics_per_seed_dict[metric + '_per_seed'][set_type]. \
                            append(np.nan)

    metrics_macro_dict = {metric + '_macro': None for metric in metrics_list}
    for metric in metrics_list:
        metrics_macro_dict[metric + '_macro'] = {'train': None, 'validation': None, 'test': None}
        for set_type in metrics_macro_dict[metric + '_macro']:
            metrics_macro_dict[metric + '_macro'][set_type] = \
                np.nanmean(metrics_per_seed_dict[metric + '_per_seed'][set_type])
    return metrics_macro_dict


if __name__ == "__main__":
    result_path = sys.argv[1]
    seed_list = [int(seed) for seed in sys.argv[2].split(',')]
    metm_dict = get_ordered_macro_metrics(result_path, seed_list)
