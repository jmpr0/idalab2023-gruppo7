import argparse
import json
import os
import sys
from functools import partial
from glob import glob
from multiprocessing import Pool

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import cm


# sns.set_style("darkgrid")


def plot_average_metrics(df, x='Episode', y='Accuracy', hue='Approach', savefig=None, hue_order=None, palette=None,
                         markers=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    sizes = df['Approach'].apply(lambda x: 'sized' if 'iCaRL' in x else 'none').values
    g = sns.lineplot(data=df, x=x, y=y, hue=hue, style=hue, hue_order=hue_order, palette=palette, markers=markers,
                     dashes=False, mec=None, ms=15, alpha=.8, clip_on=False, zorder=3, ax=ax,
                     size=sizes, sizes={'sized': 3, 'none': 0}, ci='sd',)
                     # err_style='bars', err_kws=dict(elinewidth=1, capsize=.15))
    ax.set_xticks(range(df[x].max() + 1))
    ax.set_xticklabels(['%d' % v for v in range(df[x].max() + 1)], fontsize=14)
    ax.set_yticks(range(0, 101, 10))
    ax.set_yticklabels(['%d' % v for v in range(0, 101, 10)], fontsize=14)
    ax.set_xlim(left=df[x].min(), right=df[x].max())
    ax.set_ylim(bottom=0, top=100)
    ax.set_xlabel(ax.get_xlabel(), fontsize=16)
    ax.set_ylabel('%s [%%]' % y, fontsize=16)
    h, l = g.get_legend_handles_labels()
    n_handles = len(df[hue].unique())
    leg = ax.legend(h[:n_handles], l[:n_handles],
              bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title=hue, fontsize=14, title_fontsize=16)
    leg.get_frame().set_linewidth(0.0)
    ax.grid(axis='both', which='major', ls='--', alpha=.5)
    plt.tight_layout()
    if savefig is not None:
        plt.savefig('%s.pdf' % savefig)
    plt.close(fig)


def plot_per_episode_metrics(df, x='Task', col='Episode', y='Accuracy', hue='Approach', savefig=None, hue_order=None,
                             palette=None):
    sns.set_context(rc={'patch.linewidth': 0.0})
    g = sns.catplot(data=df, x=x, col=col, y=y, hue=hue, hue_order=hue_order, palette=palette, kind='bar', legend=False,
                    col_wrap=4, ci='sd', height=3, aspect=3, errwidth=.7, capsize=.05)
    g.set(ylim=(0, 100))
    leg = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title=hue)
    leg.get_frame().set_linewidth(.0)
    for ax in g.axes.flat:
        ax.grid(axis='y', which='major', ls='--', alpha=.5)
    if savefig is not None:
        plt.savefig('%s.pdf' % savefig)
    plt.close()


def main(metric, df_filenames, exp_name, results_path, analysis, override=True):
    print('Plotting %s %s' % (analysis, metric))
    appr_dict = {'icarl': 'iCaRL-FACIL', 'icarlo': 'iCaRL-original', 'icarlp': 'iCaRL+', 'bic': 'BiC', 'il2m': 'IL2M',
                 'lwf': 'LwF', 'finetuning': 'FineTuning', 'lucir': 'LUCIR', 'ewc': 'EWC', 'joint': 'Joint',
                 'scratch': 'Scratch', 'freezing': 'Fixed Repr.'}
    metr_dict = {'accuracy_score': 'Accuracy', 'balanced_accuracy_score': 'Balanced Accuracy', 'f1_score': 'F1 Score',
                 'precision_score': 'Precision', 'recall_score': 'Recall', 'top_k_accuracy_score': 'Top k Accuracy'}

    material_path = os.path.join(results_path, exp_name, 'material')
    if not os.path.exists(material_path):
        os.makedirs(material_path)

    all_colors = [cm.get_cmap('tab20')((i + .5) / 20) for i in range(20)]
    # Markers work with sns.scatterplot
    all_markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
    hue_order = sorted(appr_dict.values())
    assert len(hue_order) <= len(all_colors), 'Not enough colors for the selected hue'
    assert len(hue_order) <= len(all_markers), 'Not enough markers for the selected hue'
    palette = dict([(h, c) for h, c in zip(hue_order, all_colors)])
    markers = dict([(h, m) for h, m in zip(hue_order, all_markers)])

    preprocessed_fn = os.path.join(material_path, '%s_metrics_%s-material.parquet' % (analysis, metric))
    if os.path.exists(preprocessed_fn) and override is None:
        override = input('File "%s" exists: override [Y, n]? ' % preprocessed_fn).lower() != 'n'
    if not os.path.exists(preprocessed_fn) or override:  # Generate or Override
        if analysis == 'average':
            df = pd.DataFrame(columns=['Seed', 'Network', 'Approach', 'Episode', metr_dict[metric]])
        else:  # if analysis == 'per_episode':
            df = pd.DataFrame(columns=['Seed', 'Network', 'Approach', 'Episode', 'Task', metr_dict[metric]])
        for f in df_filenames:
            ts = '-'.join(f.split('/')[-1].split('-')[1:]).split('_')[0]
            args_dict = json.loads(open('/'.join(f.split('/')[:-2] + ['args-%s.txt' % ts])).read())
            row = {'Seed': args_dict['seed'], 'Network': args_dict['network'],
                   'Approach': appr_dict[args_dict['approach']]}
            del args_dict
            tmp_df = pd.read_parquet(f)
            values = tmp_df[metric].values * 100
            del tmp_df
            for ep, value in enumerate(values):
                row.update({'Episode': ep})
                if analysis == 'average':
                    row.update({metr_dict[metric]: value})
                    df = df.append(row, ignore_index=True)
                else:  # if analysis == 'per_episode':
                    for task, val in enumerate(value):
                        row.update({'Task': task, metr_dict[metric]: val})
                        df = df.append(row, ignore_index=True)
        assert len(df) > 0
        df.to_parquet(preprocessed_fn)
    else:
        print('WARNING: using already computed material.')
        df = pd.read_parquet(preprocessed_fn)

    print(df)

    img_path = os.path.join(results_path, exp_name, 'img')
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    for network in df['Network'].unique():
        if analysis == 'average':
            plot_average_metrics(df[df['Network'] == network], y=metr_dict[metric],
                                 savefig=os.path.join(img_path, 'average_%s_%s' % (network, metric)),
                                 hue_order=hue_order, palette=palette, markers=markers)
        else:  # if analysis == 'per_episode':
            plot_per_episode_metrics(df[df['Network'] == network], y=metr_dict[metric],
                                     savefig=os.path.join(img_path, 'per_episode_%s_%s' % (network, metric)),
                                     hue_order=hue_order, palette=palette)


if __name__ == '__main__':
    print('IN')
    parser = argparse.ArgumentParser(description='Incremental Learning Metrics Plotter.')

    all_metrics = ['accuracy_score', 'balanced_accuracy_score', 'f1_score', 'precision_score', 'recall_score', ]
    # 'top_k_accuracy_score']

    parser.add_argument('--exp-name', default=None, type=str,
                        help='Experiment name (default=%(default)s)', nargs='+')
    parser.add_argument('--results-path', type=str, default='results',
                        help='Results path (default=%(default)s)')
    parser.add_argument('--metric', default=['f1_score'], type=str, choices=all_metrics + ['all'],
                        help='Experiment name (default=%(default)s)', nargs='*')
    parser.add_argument('--analysis', default=['average', 'per_episode'], type=str, choices=['average', 'per_episode'],
                        help='Selected analysis (default=%(default)s)', nargs='*')
    parser.add_argument('--yes', action='store_true',
                        help='Answer YES to all script requests (default=%(default)s)')
    parser.add_argument('--no', action='store_true',
                        help='Answer NO to all script requests (default=%(default)s)')
    parser.add_argument('--njobs', default=1, type=int,
                        help='Number of processes to spawn (default=%(default)s)')

    args, _ = parser.parse_known_args(sys.argv)

    assert not (args.yes and args.no), 'YES and NO cannot be set together'

    override = args.yes or not args.no if (args.yes or args.no) else None

    if 'all' in args.metric:
        args.metric = all_metrics

    for exp_name in args.exp_name:
        for analysis in args.analysis:
            df_filenames = glob('%s/*%s/**/*%s_metrics.parquet' % (args.results_path, exp_name, analysis),
                                recursive=True)

            print('WARNING: implement a mechanism that preserves the uniqueness of experiments selected.')

            # # taking only the last results for each seed
            # def get_seed_date(filename):
            #     # v = filename.split('/')[-1]
            #     v = filename
            #     seed_date = v.replace('outputs_targets_', '').replace('_%s_metrics.parquet' % analysis, '')
            #     return seed_date.split('-')[0], '-'.join(seed_date.split('-')[1:])
            #
            #
            # seed_date_dict = dict()
            # # populating a dict of list of dates for each seed
            # for df_filename in df_filenames:
            #     seed, date = get_seed_date(df_filename)
            #     seed_date_dict.setdefault(seed, []).append(date)
            # # taking only the most recent date (the last sorted) for each seed
            # for seed in seed_date_dict:
            #     seed_date_dict[seed] = sorted(seed_date_dict[seed])[-1]
            # # building the valid seed-date strings
            # latest_seed_date = ['%s-%s' % (seed, seed_date_dict[seed]) for seed in seed_date_dict]
            # # filter out not valid df_filenames basing on latest_seed_date elements
            # df_filenames = [df_filename for df_filename in df_filenames if
            #                 '%s-%s' % get_seed_date(df_filename) in latest_seed_date]

            if args.njobs <= 1:
                for metric in args.metric:
                    main(metric, df_filenames, exp_name, args.results_path, analysis, override)
            else:
                with Pool(args.njobs) as pool:
                    # N.B. override is pushed to the default value True
                    pool.map(partial(main, df_filenames=df_filenames, exp_name=exp_name,
                                     results_path=args.results_path, analysis=analysis), args.metric)
