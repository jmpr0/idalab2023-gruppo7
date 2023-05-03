from argparse import ArgumentParser
from pathlib import Path
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import datetime
import json
import os

import fsl.util.metrics 
from datasets.dataset_config import dataset_config

exp_args = [
    'train_ways', 'train_shots', 'train_queries',
    'test_ways', 'test_shots', 'test_queries', 
    'nc_train', 'approach', 'network', 'scale', 
    'loss_factor'
]
def_metrics = ['acc', 'f1', 'sc', 'db']


def compute_train_time(path):
    time_per_epoch = []

    with open(f'{path}early_stopping.log') as log:
        for i, line in enumerate(log):
            date = " ".join(line.split(' ')[:2])
            date_format = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S,%f")
            unix_time = datetime.datetime.timestamp(date_format)
            time_per_epoch.append(unix_time)
                
    return time_per_epoch[-1] - time_per_epoch[0]
    

def get_metrics(path, data, metrics, tmp):
    for folder in ['test_data', 'train_data', 'val_data']:
        last_episode = len(glob(f'{path}{folder}/logits_ep*', recursive=True))-1

        computed_metrics = fsl.util.metrics.compute_metrics(f'{path}{folder}', last_episode)
        for metric in metrics:
            data[f'{folder.replace("_data", "")}_{metric}_mean'] = computed_metrics[metric][0]
            data[f'{folder.replace("_data", "")}_{metric}_std'] = computed_metrics[metric][1]
    
    tmp.append(pd.DataFrame([data]))
    

def get_metric_dataframe(exp_path, metrics):
    tmp = []
    
    for args_path in tqdm(Path(exp_path).rglob('dict_args.json'), desc='Generating parquet'): # Get exp args
        data = dict()
        with open(args_path) as f:
            dict_args = json.load(f)
        for exp_arg in exp_args:
            data[exp_arg] = dict_args[exp_arg]
            
        if data['nc_train'] is None:
            data['nc_train'] = len(dataset_config[dict_args['dataset']]['fs_split']['train_classes'])
        
        path = str(args_path).replace('dict_args.json', '')
        
        train_time = compute_train_time(path)
        
        if 'stage_2' in path:
            max_time = float('-inf')
            for mod in [0,1]:
                mod_path = str(path).replace('stage_2', f'stage_{mod}')
                mod_data = data
                mod_data['modality'] = mod
                get_metrics(mod_path, mod_data, metrics, tmp)
                
                mod_train_time = compute_train_time(mod_path)
                if mod_train_time > max_time:
                    max_time = mod_train_time
                    
            data['modality'] = 2
            train_time += max_time 
        else:
            data['modality'] = np.nan
        
        data['train_time'] = train_time
        
        get_metrics(path, data, metrics, tmp)
            
    return pd.concat(tmp, ignore_index=True)


def main():
    parser = ArgumentParser(conflict_handler="resolve", add_help=True)

    parser.add_argument('--exp_path', type=str, default='../experiments')
    parser.add_argument('--save_path', type=str, default='../metrics')
    parser.add_argument('--name_exp', type=str, default='fsl_metrics')
    parser.add_argument(
        '--metrics', default=def_metrics, type=str, choices=def_metrics, nargs='+'
    ) 
    
    args = parser.parse_args()
    dict_args = vars(args)
    print(dict_args)

    try: 
        os.mkdir(args.save_path) 
    except OSError as error: 
        print(error)  

    df = get_metric_dataframe(args.exp_path, args.metrics)
    df.to_parquet(f'{args.save_path}/{args.name_exp}.parquet')
    df.to_csv(f'{args.save_path}/{args.name_exp}.csv')
    print('PARQUET GENERATED!!')


if __name__ == '__main__':
    main()