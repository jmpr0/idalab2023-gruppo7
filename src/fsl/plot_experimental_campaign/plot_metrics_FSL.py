from argparse import ArgumentParser
from matplotlib.ticker import MaxNLocator
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

baselines = ['freezing', 'finetuning', 'scratch']
alg_ml = ['maml', 'anil']
plots = ['bar', 'line']
metrics = [ # y-axis
    'train_acc', 'train_f1', 'train_sc',
    'test_acc', 'test_f1', 'test_sc',
    'val_acc', 'val_f1', 'val_sc'
]
exp_args = [ # x-axis
    'train_ways', 'train_shots', 'network',
    'nc_train', 'scale' #,'approach'
]
colors_dv9 = [ # Divergent Set 9 colors
    '#003f5c', '#444e86', '#955196',
    '#dd5182', '#ff6e54', '#ffa600',
    '#488f31', '#c5d275', '#2a9896'
] 
colors_dv7 = [ # Divergent Set 7 colors
    '#003f5c', '#7a5195', '#ef5675',
    '#ffa600', '#488f31', '#c5d275',
    '#2a9896'
]
colors_rs6 = [ # Rainbow Set 6 colors
    'tab:blue', 'tab:orange', 'tab:green', 
    'tab:red', 'tab:purple', 'tab:cyan'
] 
markers_9 = [
    'o', 'v', '^', 's', 'h',
    "D", 'P', 'X', '*'
]
linestyles_6 = [
    'solid', 'dotted', 'dashed', 
    'dashdot', (0, (5, 1)), (0, (3, 1, 1, 1, 1, 1))
]
appr_conv = {
    'matching_net':'MatchingNet',
    'proto_net':'ProtoNet',
    'relation_net' : 'RelationNet',
    'metaoptnet' : 'MetaOptNet',
    'maml' : 'fo-MAML',
    'anil' : 'ANIL',
    'scratch' : 'Scratch',
    'finetuning' : '$\mathtt{TL_{FT}}$',
    'freezing' : '$\mathtt{TL_{FZ}}$'
}
exp_args_conv = {
    'train_ways' : 'Ways', 
    'train_shots' : 'Shots', 
    'train_queries' : 'Queries',
    'test_ways' : 'Ways', 
    'test_shots' : 'Shots', 
    'test_queries' : 'Queries',
    'nc_train' : '# of train classes',
    'approach' : 'Approaches',
    'network' : 'Backbones',
    'scale' : 'Scale factor'
}
metrics_conv_dict = {
    'acc' : 'Accuracy', 
    'f1' : 'F1-Score', 
    'sc' : 'Silhouette Score'
}
net_conv = {
    'Wang17' : '1D-CNN',
    'Lopez17CNNRNN' : 'Hybrid', 
    'Lopez17CNN' : '2D-CNN', 
    'Aceto19MIMETIC' : 'Mimetic*'}
base_config = {
    'train_ways' : ('train_shots', 25), #25 5
    'train_shots' : ('train_ways', 8), #8 4
    #'network' : 
}


def metrics_conv(y_axis):
    metrics = ['acc', 'f1', 'sc']
    for m in metrics:
        if m in y_axis:
            return metrics_conv_dict[m]


def plot_setups(approches_total, y_axis):

    if 'val' in y_axis or 'silhouette' in y_axis: # Remove baselines from the approches
        approaches = [appr for appr in approches_total if appr not in baselines]
    else:
        approaches = approches_total
        
    if len(approaches) in [6, 8, 9]: # 6 fsl o 6 fsl + 3 baselines 
        colors = colors_dv9  
    else:
        colors = colors_dv7 # 4 fsl o 4 fsl + 3 baselines 
    n_legend_col = int(np.ceil(len(approaches)/3))
    return colors, n_legend_col, approaches


def draw_bar_plot(df, x_axis, y_axis):

    x_ticks = sorted(df[x_axis].unique())
    #x_ticks = ["Lopez17CNN", "Wang17", "Lopez17CNNRNN", "Aceto19MIMETIC"]
    #x_ticks = [x for x in x_ticks if str(x) != 'nan']
    x_ticks = [net_conv[x] for x in x_ticks]
    #x_ticks = ["Lopez17CNN"]

    #plt.rcParams.update({'font.size': 16})
    plt.rcParams["figure.figsize"] = (8,4)
    plt.rcParams['figure.dpi'] = 100

    plt.rc('axes', axisbelow=True)
    #plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(MaxNLocator(10))
    if 'silhouette_score' not in y_axis:
        plt.ylim(0, 1)
    plt.xlabel(exp_args_conv[x_axis], fontsize=16)
    plt.ylabel(metrics_conv(y_axis), fontsize=16)
    plt.grid(linestyle = '--', color="lightgray", axis='y')
    #plt.xticks(x_ticks, fontsize=14)
    plt.yticks(fontsize=14)

    approaches = df['approach'].unique()
    #approaches = ['metaoptnet', 'matchingnet', 'relationnet', 'protonet', 'anil', 'maml'] #, 'scratch', 'finetuning']
    colors, n_legend_col, approaches = plot_setups(approaches, y_axis)
    x_tick_loc = len(approaches)/2 - .5
    width = 0.09 if len(approaches) > 7 else 0.13

    for i, approach in enumerate(approaches): 
        #df[x_axis] = pd.Categorical(df[x_axis], ["Lopez17CNN", "Wang17", "Lopez17CNNRNN", "Aceto19MIMETIC"])
        temp_df = df[(df['approach'] == approach)].sort_values(x_axis)
        values = temp_df[f'{y_axis}_mean'].to_list()
        std = temp_df[f'{y_axis}_std'].to_list()
        x = np.arange(len(x_ticks))
        plt.bar(
            x-width*(x_tick_loc-i), values, label=appr_conv[approach],
            color=colors[i], width=width, yerr=std, edgecolor = 'black'
        )

    plt.xticks(x, x_ticks, fontsize=14)
    #plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.36), prop={'size': 14}, ncol=n_legend_col, frameon=False)
    plt.savefig(f'{y_axis}_on_{x_axis}.pdf', bbox_inches="tight")


def draw_line_plot(df, x_axis, y_axis):

    x_ticks = sorted(df[x_axis].unique())
    x_ticks = [x for x in x_ticks if str(x) != 'nan']

    #plt.rcParams.update({'font.size': 16})
    plt.rcParams["figure.figsize"] = (8,4) # 5.5,2.75
    plt.rcParams['figure.dpi'] = 100

    plt.rc('axes', axisbelow=True)
    #plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel(exp_args_conv[x_axis], fontsize=16)
    plt.ylabel(metrics_conv(y_axis), fontsize=16)
    plt.grid(linestyle = '--', color="lightgray")
    plt.xticks(x_ticks, fontsize=14)
    plt.yticks(fontsize=14) # np.arange(0, 0.9, 0.1),

    approaches = df['approach'].unique()
    approaches = ['metaoptnet', 'matchingnet', 'relationnet', 'protonet', 'anil', 'maml', 'finetuning', 'freezing']
    colors, n_legend_col, approaches = plot_setups(approaches, y_axis)

    for i, approach in enumerate(approaches): 
        temp_df = df[
            (df['approach'] == approach) #& 
            #(df[base_config[x_axis][0]] == base_config[x_axis][1])
        ].sort_values(x_axis) 
        if temp_df.empty:
            temp_df = df[(df['approach'] == approach)].sort_values(x_axis)

        values = temp_df[f'{y_axis}_mean'].to_list()
        std = temp_df[f'{y_axis}_std'].to_list()

        if 'way' in x_axis and (approach in baselines or approach in alg_ml):
            values = values*len(x_ticks)
            std = std*len(x_ticks)
            markers_on = [False, False, False, True]
            linestyle='dashed'
        else:
            markers_on = [True]*len(x_ticks)
            linestyle='solid'
            
        plt.plot(
            x_ticks, values, marker=markers_9[i], label=appr_conv[approach], color=colors[i],
            linewidth=2, markersize=10, markevery=markers_on, linestyle=linestyle
        )
        plt.fill_between(
            x_ticks, [m-s for m,s in zip(values, std)], [m+s for m,s in zip(values, std)],
            alpha=.15, color=colors[i]
        )

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.36), prop={'size': 14}, ncol=n_legend_col, frameon=False) # columnspacing=.6, handletextpad=.4
    plt.savefig(f'{y_axis}_on_{x_axis}.pdf', bbox_inches="tight")


def main():
    parser = ArgumentParser(conflict_handler="resolve", add_help=True)

    parser.add_argument('--metrics_path', type=str, default='./metrics/fsl_metrics.parquet')
    parser.add_argument('--plot', default='line', type=str, choices=plots)
    parser.add_argument('--x_axis', default='train_ways', type=str, choices=exp_args) 
    parser.add_argument('--y_axis', default='train_acc', type=str, choices=metrics)

    args = parser.parse_args()
    dict_args = vars(args)
    print(dict_args)

    df = pd.read_parquet(args.metrics_path)
    
    if args.plot == 'line' and args.x_axis != 'network':
        draw_line_plot(df, args.x_axis, args.y_axis)
    elif args.plot == 'bar':
        draw_bar_plot(df, args.x_axis, args.y_axis)
    else:
        print('Bad input')
        return
    print('PLOTTING METRICS DONE!!')
    

if __name__ == '__main__':
    main()