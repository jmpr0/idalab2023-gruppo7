from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
from glob import glob
import os.path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import traceback
import os
import re

import fsl.util.metrics


plot_conv = {
    'train_val' : 'Train-Validation',
    'test' : 'Test'
}
metric_conv = {
    'acc' : 'Accuracy',
    'f1' : 'F1-Score',
    'sc' : 'Silhouette Score',
    'loss' : 'Loss',
    'rec_loss' : 'Reconstruction Loss',
    'db' : 'Davies-Bouldin Score',
    'ch' : 'Calinski-Harabasz Score'
}


def load_loss(path, epoch_index):
    # Read losses
    output = dict()
    saved_losses = np.load(f'{path}/losses_ep{epoch_index}.npz')
    for loss in saved_losses.files:
        output[loss] = saved_losses[loss].mean(), saved_losses[loss].std()
    if len(output.keys()) > 1: 
        output['loss'] = tuple(
            map(lambda i, j: i - j, output['loss'], output['rec_loss'])
        ) # 'loss' = ce_loss + rec_loss -> 'loss' = ce_loss
    return output


def plot_metrics(title, label, y_label, f1, f2, max_acc_epoch, save_path):
    
    plt.rcParams.update({'font.size': 14})
    plt.rcParams["figure.figsize"] = (10,5)
    plt.rcParams['figure.dpi'] = 100

    plt.rc('axes', axisbelow=True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Epoch')
    plt.ylabel(y_label)
    plt.title(title) 
    plt.grid(linestyle = '--', color="lightgray")
    
    meta_phase = 'Test' if 'Test' in label else 'Train'
    f1_mean = [i[0] for i in f1]
    f1_std = [i[1] for i in f1]
    x = list(range(0, len(f1_mean))) 
    
    plt.axvline(x=int(max_acc_epoch), color='black', linestyle='--')
    plt.plot(x, f1_mean, marker='o', label=f'{meta_phase}', color='#FF8484')
    plt.fill_between(
        x, [m-s for m,s in zip(f1_mean, f1_std)],[m+s for m,s in zip(f1_mean, f1_std)], 
        alpha=0.15, color='#FF8484'
    )
    
    if f2 is not None:
        f2_mean = [i[0] for i in f2]
        f2_std = [i[1] for i in f2]
        plt.plot(x, f2_mean, marker='s', label=f'Validation', color='#437AB6')
        plt.fill_between(
            x, [m-s for m,s in zip(f2_mean, f2_std)], [m+s for m,s in zip(f2_mean, f2_std)],
            alpha=0.15, color='#437AB6'
        )
    plt.legend()
    plt.savefig(f'{save_path}/{label}.pdf', bbox_inches="tight")
    plt.clf()
   

def write_metrics_parquet(data, save_path):
    n_epochs = len(data['train']['acc'])
    tmp = []
    for epoch in range(0, n_epochs):
        row = dict()
        row['epoch'] = epoch
        for phase in ['train', 'test', 'val']:
            for metric in ['acc', 'f1', 'sc', 'loss']:
                row[f'{phase}_{metric}_mean'] = data[phase][metric][epoch][0]
                row[f'{phase}_{metric}_std'] = data[phase][metric][epoch][1]
        tmp.append(pd.DataFrame([row]))
    df = pd.concat(tmp, ignore_index=True)
    df.to_parquet(f'{save_path}/per_epoch_metrics.parquet')
        

def plot_experiment_log(exp_path):
    data = {'test':{}, 'train':{}, 'val':{}}
    
    if os.path.isfile(f'{exp_path}/dict_args.json'):
        dict_args_path = f'{exp_path}/dict_args.json'
    else: 
        # TODO: use regex
        to_replace = 'stage_0' if 'stage_0' in exp_path else 'stage_1'
        dict_args_path = f"{exp_path.replace(to_replace, 'stage_2')}/dict_args.json"
    # Get usefull experiment args
    with open(dict_args_path) as f: 
        dict_args = json.load(f) 
        
    # Epoch where max val acc has been achieved
    x_max = re.findall(r'\d+', glob(f'{exp_path}/checkpoints/*')[0])[-1]
    
    for folder in tqdm(['test_data', 'train_data', 'val_data'], desc=f'Computing metrics'):
        
        # Setting useful parameters
        meta_phase = f'{folder.replace("_data", "")}'
        n_files = len(glob(f'{exp_path}/{folder}/*')) 
                                
        for i in range(n_files//6):  # For each epoch compute metrics & losses # TODO: remove 6 - use an automatic mechanism
            computed_metrics = fsl.util.metrics.compute_metrics(f'{exp_path}/{folder}', i)
            losses = load_loss(f'{exp_path}/{folder}', i)
            for key, value in {**computed_metrics, **losses}.items():
                data[meta_phase].setdefault(key, []).append(value)
            
    # Write a parquet that logs for each epoch the performance  
    write_metrics_parquet(data, exp_path)
    
    try:
        os.makedirs(f'{exp_path}/img')
        imgs_path = f'{exp_path}/img'
    except FileExistsError:
        print(traceback.format_exc())
        return
    
    # Plot
    metrics_to_plot = ['acc', 'sc', 'loss', 'db', 'ch']
    if 'rec_loss' in data['train']:
        metrics_to_plot.extend(['rec_loss'])
    for metric_to_plot in tqdm(metrics_to_plot, desc=f'Plotting'): # Add here the functions to be plotted
        for plot_type in ['train_val', 'test']: # Two types of plot: test and train+val
            if plot_type == 'test':
                f1 = data['test'][metric_to_plot]
                f2 = None
            elif plot_type == 'train_val':
                f1 = data['train'][metric_to_plot]
                f2 = data['val'][metric_to_plot]
            else:
                print('Missing plot type')
                return
            plot_label = f'{plot_conv[plot_type]} {metric_conv[metric_to_plot]}'
            title = f'{dict_args["approach"]}-n-{dict_args["train_ways"]}-k-{dict_args["train_shots"]}'
            plot_metrics(
                title=title, label=plot_label,
                y_label=metric_conv[metric_to_plot],f1=f1, f2=f2,
                max_acc_epoch=x_max, save_path=imgs_path
            )