import argparse
import functools
import hashlib
import itertools
import os
import sys
from copy import deepcopy
from functools import partial
from glob import glob
from multiprocessing import Pool
from time import sleep

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import portalocker as pl
import seaborn as sns
from matplotlib import cm
from tqdm import tqdm
from util.colour_mapper import colour_mapper
from util.metrics_utils import *

mpl.use('Agg')


def plot_average_metrics(df, x='Episode', y='Accuracy', hue='Approach', savefig=None, hue_order=None, palette=None,
                         markers=None, ylim='auto', plot_type='lineplot', no_title=False, figsize=(6.5, 3.3)):
    # TODO: extract hardcoded values
    fig, ax = plt.subplots(figsize=figsize)
    # styles = df['Approach'].apply(lambda x: 'styled' if x in ['LwF', 'Fixed-Repr', 'FineTuning'] else 'none').values
    # styles = df['Approach'].apply(lambda x: 'styled').values
    dashes = dict([(h, (5, 10)) if h in ['LwF', 'Frozen-BB', 'FineTuning', 'Joint-Train', 'Feat-Extr']
                   else (h, '') for h in hue_order])
    # print(markers)
    if plot_type == 'lineplot':
        g = sns.lineplot(data=df, x=x, y=y, hue=hue, style=hue, hue_order=hue_order, palette=palette, markers=markers,
                         dashes=dashes, mec=None, ms=15, alpha=.8, clip_on=False, zorder=3, ax=ax, ci='sd')
        ax.set_xticks(range(df[x].max() + 1))
        xticklabels = ['%d' % v for v in range(df[x].max() + 1)]
        ax.set_xticklabels(
            xticklabels if (df[x].max() - df[x].min()) < 10
            else [v if not i % 2 else '' for i, v in enumerate(xticklabels)],
            fontsize=14)
        print(dict(left=df[x].min(), right=df[x].max()))
        ax.set_xlim(left=df[x].min(), right=df[x].max())
    elif plot_type == 'barplot':
        df = df[df['Episode'] == 1]
        g = sns.barplot(data=df, x=x, y=y, hue=hue, hue_order=hue_order, palette=palette, ax=ax, ci='sd')
    # styles={'styled': 3, 'none': 1}, )
    # err_style='bars', err_kws=dict(elinewidth=1, capsize=.15))
    ax.yaxis.set_tick_params(labelsize=14)
    if ylim != 'auto':
        y_min = df[y].min()
        if y_min < 0:
            bottom = int(-np.ceil(-y_min / 10) * 10)
        else:
            bottom = 0
        ax.set_yticks(range(bottom, 101, 10))
        ax.set_yticklabels(['%d' % v for v in range(bottom, 101, 10)], fontsize=14)
        ax.set_ylim(bottom=bottom, top=100)
    ax.set_xlabel(ax.get_xlabel(), fontsize=16)
    ax.set_ylabel('%s [%%]' % y, fontsize=16)
    h, l = g.get_legend_handles_labels()
    n_handles = len(df[hue].unique())
    leg = ax.legend(h[:n_handles], l[:n_handles],
                    bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title=hue, fontsize=14, title_fontsize=16, )
    # ncol=2)
    leg.get_frame().set_linewidth(.0)
    ax.grid(axis='both', which='major', ls='--', alpha=.5)
    infos = ['Seed', 'Approach', 'Batch Size', 'Patience']
    if not no_title:
        title = ''
        for info in infos:
            if info == hue:
                continue
            title += '%s: %s    ' % (
                info, ', '.join(['%s' % (int(v) if isinstance(x, float) else v) for v in sorted(df[info].unique())]))
        ax.set_title(title, fontsize=16)
    if savefig is not None:
        plt.tight_layout()
        plt.savefig('%s%s.pdf' % (savefig, '' if no_title else '_draft'))
        plt.savefig('%s%s.png' % (savefig, '' if no_title else '_draft'), dpi=300)
        print('Plotted at %s.pdf' % savefig)
    plt.close(fig)


def plot_per_episode_metrics(df, x='Task', col='Episode', y='Accuracy', hue='Approach', savefig=None, hue_order=None,
                             palette=None, no_title=False):
    sns.set_context(rc={'patch.linewidth': 0.0})
    g = sns.catplot(data=df, x=x, col=col, y=y, hue=hue, hue_order=hue_order, palette=palette, kind='bar', legend=False,
                    col_wrap=4, ci='sd', height=3, aspect=2, errwidth=.7, capsize=.05)
    g.set(ylim=(0, 100))
    leg = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title=hue)
    leg.get_frame().set_linewidth(.0)
    for ax in g.axes.flat:
        ax.grid(axis='y', which='major', ls='--', alpha=.5)
    infos = ['Seed', 'Approach', 'Batch Size', 'Patience']
    if not no_title:
        title = ''
        for info in infos:
            if info == hue:
                continue
            title += '%s: %s    ' % (
                info, ', '.join(['%s' % (int(v) if isinstance(x, float) else v) for v in sorted(df[info].unique())]))
        g.fig.suptitle(title, fontsize=16)
    if savefig is not None:
        plt.tight_layout()
        plt.savefig('%s%s.pdf' % (savefig, '' if no_title else '_draft'))
        plt.savefig('%s%s.png' % (savefig, '' if no_title else '_draft'), dpi=300)
        print('Plotted at %s.pdf' % savefig)
    plt.close()


def plot_confusion_matrix(df_filenames, plot_path, exp_name, std=False):
    print('Plotting %s confusion matrix.' % df_filenames)
    sns.set(font='DejaVu Sans Mono')

    single_seed = False
    if isinstance(df_filenames, str):
        df_filenames = [df_filenames]
        single_seed = True

    n_seeds = len(df_filenames)
    args_dict = get_args_dict(df_filenames[-1])

    if single_seed:
        img_path = os.path.join(plot_path, 'img', 'confusion_matrix', args_dict['approach'], str(args_dict['seed']))
    else:
        img_path = os.path.join(plot_path, 'img', 'confusion_matrix', args_dict['approach'], 'average')

    with pl.Lock('lock'):
        if not os.path.exists(img_path):
            os.makedirs(img_path)

    confusion_matrices = []
    for df_filename in df_filenames:
        df = pd.read_parquet(df_filename)
        confusion_matrices.append(df['confusion_matrix'].values)

    num_base_classes = args_dict.get('nc_first_task', len(confusion_matrices[0][0]))
    for ep in range(len(confusion_matrices[0])):
        fig, ax = plt.subplots()
        confusion_matrix_1, rec, prec, f1 = [], [], [], []
        new_rec, new_prec, new_f1 = [], [], []
        for i in range(n_seeds):
            confusion_matrix = np.array([[v1 for v1 in v] for v in confusion_matrices[i][ep]], dtype='float')
            # CM with Precisions on the diagonal
            sum_ax0 = np.sum(confusion_matrix, axis=0)
            sum_ax0[sum_ax0 == 0] = np.inf
            confusion_matrix_0 = confusion_matrix / sum_ax0
            # CM with Recalls on the diagonal
            sum_ax1 = np.sum(confusion_matrix, axis=1).reshape(-1, 1)
            sum_ax1[sum_ax1 == 0] = np.inf
            confusion_matrix_1.append(confusion_matrix / sum_ax1)

            rec.append(np.mean(np.diagonal(confusion_matrix_1[-1])[:num_base_classes]) * 100)
            prec.append(np.mean(np.diagonal(confusion_matrix_0)[:num_base_classes]) * 100)
            f1.append(2 * rec[-1] * prec[-1] / (rec[-1] + prec[-1]) if rec[-1] + prec[-1] != 0 else 0)
            if ep > 0:
                new_rec.append(np.mean(np.diagonal(confusion_matrix_1[-1])[num_base_classes:]) * 100)
                new_prec.append(np.mean(np.diagonal(confusion_matrix_0)[num_base_classes:]) * 100)
                new_f1.append(2 * new_rec[-1] * new_prec[-1] / (new_rec[-1] + new_prec[-1]) if new_rec[-1] + new_prec[
                    -1] != 0 else 0)

        confusion_matrix_1 = np.mean(confusion_matrix_1, axis=0)

        num_classes = confusion_matrix_1.shape[0]
        sns.heatmap(confusion_matrix_1, vmin=.0, vmax=1., cmap='Reds', square=True, cbar=False, ax=ax, zorder=3)
        if ep > 0:
            for fn in [ax.vlines, ax.hlines]:
                fn(num_base_classes, 0, num_classes, linestyles='dashed', color='royalblue', zorder=4)
        ax.set_xticks([v + .5 for v in range(num_classes)])
        ax.set_xticklabels([v if i % 2 == 0 else '' for i, v in enumerate(range(num_classes))], rotation=0)
        # ax.set_yticks([v + .5 for v in range(num_classes)])
        # ax.set_yticklabels([v if i % 2 == 0 else '' for i, v in enumerate(range(num_classes))], rotation=0)
        ax.xaxis.set_tick_params(width=5, color='k', labelsize=14)
        ax.yaxis.set_tick_params(width=5, color='k', labelsize=14)
        trans = ax.get_yaxis_transform()  # y in data untis, x in axes fraction
        if std:
            ax.annotate('Base Apps\nRec  = %2.2f$\pm$%2.2f%%\nPrec = %2.2f$\pm$%2.2f%%\nF1   = %2.2f$\pm$%2.2f%%' % (
                np.mean(rec), np.std(rec), np.mean(prec), np.std(prec), np.mean(f1), np.std(f1)),
                        xy=(1.02, num_classes * .2), xycoords=trans, zorder=5)
        else:
            ax.annotate('Base Apps\nRec  = %2.2f%%\nPrec = %2.2f%%\nF1   = %2.2f%%' % (
                np.mean(rec), np.mean(prec), np.mean(f1)),
                        xy=(1.02, num_classes * .2), xycoords=trans, zorder=5)
        if ep > 0:
            if std:
                ax.annotate(
                    'New Apps\nRec  = %2.2f$\pm$%2.2f%%\nPrec = %2.2f$\pm$%2.2f%%\nF1   = %2.2f$\pm$%2.2f%%' % (
                        np.mean(new_rec), np.std(new_rec), np.mean(new_prec), np.std(new_prec), np.mean(new_f1),
                        np.std(new_f1)),
                    xy=(1.02, num_classes * .8), xycoords=trans, zorder=5)
            else:
                ax.annotate('New Apps\nRec  = %2.2f%%\nPrec = %2.2f%%\nF1   = %2.2f%%' % (
                    np.mean(new_rec), np.mean(new_prec), np.mean(new_f1)),
                            xy=(1.02, num_classes * .8), xycoords=trans, zorder=5)
        ax.set_title(exp_name, fontsize=16)

        savefig = os.path.join(img_path, 'confusion_matrix_%s_ep%s_%s_%s' % (
            args_dict['approach'], ep, args_dict['network'], 'sd%s' % args_dict['seed'] if single_seed else 'average'))
        plt.tight_layout()
        saved = False
        while not saved:
            try:
                plt.savefig('%s.pdf' % savefig)
                plt.savefig('%s.png' % savefig, dpi=300)
                plt.close(fig)
                print('Plotted at %s.pdf' % savefig)
                saved = True
            except PermissionError as pe:
                print('%s for %s.' % (pe, df_filenames))
                sleep(np.random.randint(0, 10))

        num_base_classes = confusion_matrix_1.shape[0]


def plot_training(df_train, df_timing, plot_path, exp_name, std=False):
    img_path = os.path.join(plot_path, 'img', 'training')
    with pl.Lock('lock'):
        if not os.path.exists(img_path):
            os.makedirs(img_path)
    df_train = df_train.astype({'Seed': int})
    df_train = df_train.astype({'Task': int})
    sns.relplot(data=df_train, x='Epoch', y='Validation Loss', col='Task', height=4, aspect=.7, kind='line',
                color='orange')
    plt.savefig(img_path + '/ValidationLoss_tasks.pdf')
    sns.relplot(data=df_train, x='Epoch', y='TAw Accuracy', col='Task', height=4, aspect=.7, kind='line')
    plt.savefig(img_path + '/TAw_tasks.pdf')
    for t in np.unique(df_train.Task.tolist()):
        df_t = df_train[(df_train.Task == t)]
        seeds = ' '.join([str(elem) for elem in np.unique(df_t['Seed'].tolist())])
        df_group = df_t.groupby('Epoch').describe()
        fig, ax = plt.subplots()
        df_group['TAw Accuracy']['mean'].plot(kind='line', yerr=df_group['TAw Accuracy']['std'], alpha=0.5, ax=ax,
                                              label='Taw Accuracy', legend=True)
        df_group['Validation Loss']['mean'].plot(kind='line', yerr=df_group['Validation Loss']['std'], alpha=0.5, ax=ax,
                                                 secondary_y=True, label='Val Loss', legend=True)
        lines = ax.get_lines() + ax.right_ax.get_lines()
        plt.legend(lines, [l.get_label() for l in lines], bbox_to_anchor=(0., 1.05, 1., .102), loc='center',
                   ncol=8, frameon=False, handlelength=1.50, columnspacing=1.5, fontsize=13, handletextpad=0.25,
                   numpoints=0.5)
        plt.title('Task: %s, Seed: %s' % (t, seeds))
        plt.savefig(img_path + '/Task %s.pdf' % t)
        for s in np.unique(df_t['Seed'].tolist()):
            fig, ax = plt.subplots()
            df_s = df_t[(df_t.Seed == s)]
            df_s.plot(x='Epoch', y='TAw Accuracy', ax=ax, legend=False)
            df_s.plot(x='Epoch', y='Validation Loss', ax=ax, secondary_y=True, legend=False)
            lines = ax.get_lines() + ax.right_ax.get_lines()
            plt.legend(lines, [l.get_label() for l in lines], bbox_to_anchor=(0., 1.05, 1., .102), loc='center',
                       ncol=8, frameon=False, handlelength=1.50, columnspacing=1.5, fontsize=13, handletextpad=0.25,
                       numpoints=0.5)
            plt.title('Task: %s, Seed: %s' % (t, s))
            plt.savefig(img_path + '/Task: %s, Seed: %s.pdf' % (t, s))
            plt.close()


def plot_analysis(metric_df, metr_dict, mety_dict, plot_path, analysis, x='Episode', pivots=None, hue='Approach',
                  hue_order=None, no_ylim=False, plot_type='lineplot', no_title=False, appr_dict=None):
    metric, metric_type = metric_df[0]
    df = deepcopy(metric_df[1])

    # print(df['Approach'].unique())
    # print(appr_dict)
    # input()
    # df.loc[:, 'Approach'] = df['Approach'].apply(lambda x: appr_dict.get(x, ''))

    print('Plotting %s %s %s' % (analysis, metric, metric_type))

    if pivots is None:
        pivots = ['Network']
    if '#Apps' in df:
        x = '#Apps'
        df[x] = df[x].astype(int)

    analysis_discrs = ['All', 'Base'] if metric_type == 'normal' else ['Old', 'New']

    for analysis_discr in analysis_discrs:

        discr = mety_dict[metric_type][analysis_discr.lower()]
        metric_col = ' '.join([metr_dict[metric], discr]).strip()

        _df = df.loc[df['Type'] == analysis_discr, :]
        _df.rename(columns={metr_dict[metric]: metric_col}, inplace=True)

        n_colors = 20
        all_colors = [cm.get_cmap('tab%s' % n_colors)((i + .5) / n_colors) for i in range(n_colors)]
        # Markers work with sns.scatterplot
        all_markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h',
                       'H', 'D', 'd', 'P', 'X', 'o', 'v', '^', '<', '>']
        if hue_order is not None:
            assert len(_df[hue].unique()) <= len(all_colors),\
                'Not enough colors for the selected hue: %s' % _df[hue].unique()
            assert len(_df[hue].unique()) <= len(all_markers),\
                'Not enough markers for the selected hue: %s' % _df[hue].unique()
            _df = _df[_df['Approach'].isin(appr_filt)]
            palette = dict([(h, c) for h, c in zip(hue_order, all_colors)])
            markers = dict([(h, m) for h, m in zip(hue_order, all_markers)])
        if no_ylim:
            ylim = 'auto'
        else:
            ylim = 'expand'

        img_path = os.path.join(plot_path, 'img', analysis)
        with pl.Lock('lock'):
            if not os.path.exists(img_path):
                os.makedirs(img_path)

        for pivot_values in itertools.product(*[_df[pivot].unique() for pivot in pivots]):
            print('Pivoting:', pivot_values)
            df_filter = functools.reduce(
                lambda x, y: x & y, [_df[pivot] == pivot_value for pivot, pivot_value in zip(pivots, pivot_values)])
            if hue_order is not None:
                approaches = _df[df_filter]['Approach'].unique()
                hue_order_ = [approach for approach in hue_order if approach in approaches]
            else:
                hue_order_ = sorted(_df[df_filter][hue].unique())
                palette = dict([(h, c) for h, c in zip(hue_order_, all_colors)])
                markers = dict([(h, m) for h, m in zip(hue_order_, all_markers)])
            # palette_ = dict([(h, palette[h]) for h in hue_order_])
            palette_ = dict([(appr_dict[k], v) for k, v in colour_mapper().items() if k in appr_disk_filt])
            markers_ = dict([(h, markers[h]) for h in hue_order_])
            if not palette_ and not markers_:
                palette_ = markers_ = None
            # Selecting common seed for each hue, in order to properly assess performance
            # print(_df.columns)
            # input()
            df_g = _df[(df_filter) & ~(_df[metric_col].isna())].groupby([x, hue])
            common_seeds = None
            for g in df_g:
                if g[0][1].lower() == 'scratch' or g[0][1].lower() == 'joint':
                    continue
                if common_seeds is None:
                    common_seeds = list(g[1]['Seed'].unique())
                else:
                    common_seeds = [v for v in common_seeds if v in g[1].values]
                # print(g[1])
                # input()
                # print(g[0], common_seeds)
                # input()
            if common_seeds is None:
                continue
            if 0 in common_seeds and len(common_seeds) > 1:
                del common_seeds[common_seeds.index(0)]
            print('Common seeds:', list(sorted(common_seeds)))
            if len(common_seeds) < 10 and common_seeds != [0]:
                print([(g[0], g[1]['Seed'].unique()) for g in df_g if len(g[1]['Seed'].unique()) < 10])
                # input()
            # input()
            if common_seeds is None or not len(common_seeds):
                return
            if analysis == 'per_episode':
                plot_per_episode_metrics(
                    _df[df_filter & (_df['Seed'].isin(common_seeds))], y=metric_col,
                    savefig=os.path.join(img_path, 'per_episode_%s' % '_'.join(
                        [str(v) for v in pivot_values] + [
                            '%s%s' % (metric, (discr if discr else ' ').split(' ')[0].lower())])),
                    hue=hue, hue_order=hue_order_, palette=palette_, no_title=no_title)
            else:
                # print(_df[df_filter & (_df['Seed'].isin(common_seeds))])
                # input()
                plot_average_metrics(
                    _df[df_filter & (_df['Seed'].isin(common_seeds))], x=x, y=metric_col,
                    savefig=os.path.join(img_path, '%sp_average_%s_%s' % (plot_type[0], analysis, '_'.join(
                        [str(v) for v in pivot_values] + [
                            '%s%s' % (metric, (discr if discr else ' ').split(' ')[0].lower())]))),
                    hue=hue, hue_order=hue_order_, palette=palette_, markers=markers_, ylim=ylim,
                    plot_type=plot_type, no_title=no_title)


def preprocess_metrics(metric, mety_dict, df_filenames, df_filenames_ub, plot_path, analysis, override=True):
    metric, metric_type = metric
    print('Preprocessing %s %s %s' % (analysis, metric, metric_type))

    material_path = os.path.join(plot_path, 'material')
    with pl.Lock('lock'):
        if not os.path.exists(material_path):
            os.makedirs(material_path)

    discr = mety_dict[metric_type][analysis]

    preprocessed_fn = os.path.join(material_path,
                                   '%s_metrics_%s_%s-material.parquet' % (analysis, metric, discr.lower()))
    if os.path.exists(preprocessed_fn) and override is None:
        override = input('File "%s" exists: override [Y, n]? ' % preprocessed_fn).lower() != 'n'
    if not os.path.exists(preprocessed_fn) or override:  # Generate or Override
        df, scenarios = load_data(df_filenames, metric, return_scenarios=True,
                                  per_episode=analysis=='per_episode_metrics')
        assert len(df) > 0

        # print(scenarios)
        # input()

        df_scratch = pd.DataFrame()
        if len(df_filenames_ub):
            for scenario in scenarios:
                df_scratch = df_scratch.append(load_scratch_data(df_filenames_ub, metric, *scenario),
                                               ignore_index=True)

        # if -1 not in df['Last App']:
        #     df = df[df['Seed'] != 0]
        # if -1 not in df_scratch['Last App']:
        #     df_scratch = df_scratch[df_scratch['Seed'] != 0]

        # print(list(sorted(df['Last App'].unique())), list(sorted(df['Seed'].unique())))
        # print(list(sorted(df_scratch['Last App'].unique())), list(sorted(df_scratch['Seed'].unique())))
        # input()
        #
        # print(df.groupby('Approach')['Seed'].unique().apply(lambda x: len(x)))
        # input()
        # print(df_scratch.groupby('Approach')['Seed'].unique().apply(lambda x: len(x)))
        # input()

        if metric_type == 'drop':
            df = compute_drops(df, df_scratch, metric)
        else:
            df = df.append(df_scratch, ignore_index=True)
        df.to_parquet(preprocessed_fn)
    else:
        print('WARNING: using already computed material.')
        df = pd.read_parquet(preprocessed_fn)

    if len(df['Approach'].unique()) > 1 and '#Apps' in df:
        min_classes = df[df['Approach'] != 'scratch']['#Apps'].min()
        df = df[df['#Apps'] >= min_classes]

    print(df.columns)
    print(df)
    print(df['Approach'].unique())
    return df


# INFO: the order of classes does not represent the original (sorted) one
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Incremental Learning Metrics Plotter.')

    all_metrics = ['accuracy_score', 'balanced_accuracy_score', 'f1_score', 'precision_score', 'recall_score']
    all_metrics = all_metrics + ['forgetting_%s' % v for v in all_metrics]
    all_metric_types = ['normal', 'drop']
    all_analysis = ['per_class_metrics', 'per_episode_metrics', 'confusion_matrix', 'training']
    all_classes = ['all', 'base', 'old', 'new']

    parser.add_argument('--exp-name', default=None, type=str,
                        help='Experiment name (default=%(default)s)', nargs='+')
    parser.add_argument('--partial-exp-name', action='store_true',
                        help='If the exp-name should match as *exp-name* (default=%(default)s)')
    parser.add_argument('--results-path', type=str, default='../results',
                        help='Results path (default=%(default)s)')
    parser.add_argument('--metric', default=['f1_score'], type=str, choices=all_metrics + ['all'],
                        help='Experiment name (default=%(default)s)', nargs='*')
    parser.add_argument('--metric-type', default=['normal'], type=str, choices=all_metric_types + ['all'],
                        help='Experiment name (default=%(default)s)', nargs='*')
    parser.add_argument('--analysis', default=['all'], type=str, choices=all_analysis + ['all'],
                        help='Selected analysis (default=%(default)s)', nargs='*')
    parser.add_argument('--class', default=['all'], type=str, choices=all_classes,
                        help='Selected set of classes (default=%(default)s)', nargs='*')
    parser.add_argument('--yes', action='store_true',
                        help='Answer YES to all script requests (default=%(default)s)')
    parser.add_argument('--no', action='store_true',
                        help='Answer NO to all script requests (default=%(default)s)')
    parser.add_argument('--njobs', default=1, type=int,
                        help='Number of processes to spawn (default=%(default)s)')
    parser.add_argument('--hue', type=str, default='approach',
                        help='Hue value for lineplots and barplots (default=%(default)s)')
    parser.add_argument('--pivot', default=['network'], type=str,
                        help='Pivot value for lineplots and barplots (default=%(default)s)', nargs='*')
    parser.add_argument('--nc-incr-tasks', default=None, type=int,
                        help='Bugfix for missing --nc-incr-tasks (default=%(default)s)', nargs='+')
    parser.add_argument('--no-upperbound', action='store_true',
                        help='Avoid the upperbound loading (default=%(default)s)')
    parser.add_argument('--stop-at-task', default=-1, type=int,
                        help='Number of task to stop (default=%(default)s)')
    parser.add_argument('--no-ylim', action='store_true',
                        help='Does not force the ylim in (0,100) (default=%(default)s)')
    parser.add_argument('--no-discard', action='store_true',
                        help='Does not discard duplicates (default=%(default)s)')
    parser.add_argument('--plot-type', default='lineplot', type=str, choices=['lineplot', 'barplot'],
                        help='Type of the plot between lineplot and barplot (default=%(default)s)')
    parser.add_argument('--no-title', action='store_true',
                        help='Does not show the title (default=%(default)s)')
    parser.add_argument('--paper-plot', default=None, type=str,
                        choices=['ranking', 'naive', 'lineplot', 'lineplot_net'],
                        help='Type of the plot used in the paper (default=%(default)s)')

    args, _ = parser.parse_known_args(sys.argv)

    exp_name_discr = '*' if args.partial_exp_name else ''

    if args.nc_incr_tasks is None:
        args.nc_incr_tasks = [0] * len(args.exp_name)

    upperbound_results_path = None
    if not args.no_upperbound:
        _upperbound_results_path = '%s_UB' % args.results_path.rstrip('/')
        if os.path.exists(_upperbound_results_path):
            upperbound_results_path = _upperbound_results_path
        else:
            print('WARNING: metrics from upperbound cannot be computed. Missing folder: %s' % _upperbound_results_path)

    # results_path_nas = '/media/nas/datasets/MIRAGE_2020/FSCIL_approaches/hf-project/results'
    # print('WARNING: the results path is forced to on the NAS dir %s.' % results_path_nas)
    # args.results_path = results_path_nas

    assert not (args.yes and args.no), 'YES and NO cannot be set together'

    override = args.yes or not args.no if (args.yes or args.no) else None

    if args.njobs > 1 and override is None:
        override = False

    appr_disk_filt = get_approaches(args.paper_plot)
    appr_filt = [appr_dict[appr] for appr in appr_disk_filt]

    print('\nConsidered approaches:')
    print(appr_filt)
    print(appr_disk_filt, '\n')
    sleep(2)

    if 'all' in args.metric:
        args.metric = all_metrics
    if 'all' in args.metric_type:
        args.metric_type = all_metric_types
    if 'all' in args.analysis:
        args.analysis = all_analysis

    plot_cm = 'confusion_matrix' in args.analysis
    if plot_cm:
        del args.analysis[args.analysis.index('confusion_matrix')]
    plot_train = 'training' in args.analysis
    if plot_train:
        del args.analysis[args.analysis.index('training')]

    exp_hash = hashlib.md5(''.join(sorted(args.exp_name)).encode()).hexdigest()[:10]
    plot_path = os.path.join(
        args.results_path, 'images',
        '%s_%s' % (
            ''.join([''.join(['%s%s' % (v0[:2], v0[-2:]) for v0 in v.split('_')]) for v in sorted(args.exp_name)]),
            exp_hash))
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    with open(os.path.join(plot_path, 'exp_name.info'), 'w') as fout:
        fout.writelines([exp_name + '\n' for exp_name in args.exp_name])

    args.hue = cols_dict[args.hue]
    args.pivot = [cols_dict[pivot] for pivot in args.pivot]
    hue_order = [v for v in sorted(appr_dict.values()) if v in appr_filt] if args.hue == 'Approach' else None

    for analysis in args.analysis:
        df_filenames = []
        df_filenames_ub = []
        for i, exp_name in enumerate(args.exp_name):
            df_filename = glob('%s/*%s%s/**/*%s.parquet' % (args.results_path, exp_name, exp_name_discr, analysis),
                               recursive=True)
            if len(df_filename) == 0:
                print('WARNING: no files found for --exp-name=%s' % exp_name)
            df_filenames.extend(df_filename)
        if upperbound_results_path is not None:
            df_filenames_ub = glob('%s/*%s*/**/*%s.parquet' % (upperbound_results_path, 'scratch', analysis),
                                  recursive=True)

        df_filenames = [df_filename for df_filename in df_filenames if
                        sum([f'_{af}_' in df_filename for af in appr_disk_filt])]
        if not args.no_discard:
            df_filenames = discard_duplicates(df_filenames, take_newest=True)

        metrics = [(m, t) for m in args.metric for t in args.metric_type]

        hue = args.hue
        pivot = args.pivot
        _hue_order = hue_order
        # if 'old_new' in analysis and (hue != 'Type' or 'Approach' not in pivot):
        #     print('WARNING: hue or pivot will be override for old_new analysis')
        #     hue = 'Type'
        #     pivot = pivot + ['Approach']
        #     _hue_order = None
        if 'average_metrics' in analysis:
            _analysis = analysis.split('_')[2:]
        else:
            _analysis = [analysis]

        if args.njobs <= 1:
            for metric in metrics:
                for a in _analysis:
                    df = preprocess_metrics(metric, mety_dict, df_filenames, df_filenames_ub, plot_path, a,
                                            override)
                    if args.stop_at_task > -1:
                        df = df[df['Episode'] <= args.stop_at_task]
                    plot_analysis((metric, df), metr_dict, mety_dict, plot_path, a,
                                  pivots=pivot, hue=hue, hue_order=_hue_order, no_ylim=args.no_ylim,
                                  plot_type=args.plot_type, no_title=args.no_title, appr_dict=appr_dict)
        else:
            with Pool(args.njobs) as pool:
                for a in _analysis:
                    dfs = pool.map(partial(preprocess_metrics, mety_dict=mety_dict,
                                           df_filenames=df_filenames, df_filenames_ub=df_filenames_ub,
                                           plot_path=plot_path, analysis=a, override=override),
                                   metrics)
                    for i in range(len(dfs)):
                        if args.stop_at_task > 0:
                            dfs[i] = dfs[i][dfs[i]['Episode'] <= args.stop_at_task]
                    # N.B. override is pushed to the default value True
                    pool.map(partial(plot_analysis, metr_dict=metr_dict, mety_dict=mety_dict, plot_path=plot_path,
                                     analysis=a, pivots=pivot, hue=hue, hue_order=_hue_order, no_ylim=args.no_ylim,
                                     plot_type=args.plot_type, no_title=args.no_title, appr_dict=appr_dict),
                             zip(metrics, dfs))

    if plot_train:

        for exp_name in args.exp_name:
            txt_filenames = glob('%s/*%s%s/**/*.txt' % (args.results_path, exp_name, exp_name_discr),
                                 recursive=True)
            stdout_filenames = [f for f in txt_filenames if 'stdout' in f]

            if not args.no_discard:
                df_filenames = discard_duplicates(stdout_filenames, take_newest=True)

            df_train, df_timing = get_training_info(stdout_filenames)
            if args.njobs <= 1:
                plot_training(df_train, df_timing, plot_path, exp_name)
            else:
                with Pool(args.njobs) as pool:
                    pool.map(partial(plot_training, plot_path=plot_path, exp_name=exp_name),
                             df_train, df_timing)

    if plot_cm:
        for exp_name in args.exp_name:
            df_filenames = glob('%s/*%s%s/**/*%s.parquet' % (
                args.results_path, exp_name, exp_name_discr, 'average_metrics_all'), recursive=True)

            if not args.no_discard:
                df_filenames = discard_duplicates(df_filenames, take_newest=True)

            if args.njobs <= 1:
                for df_filename in df_filenames:
                    plot_confusion_matrix(df_filename, plot_path, exp_name)
            else:
                with Pool(args.njobs) as pool:
                    pool.map(partial(plot_confusion_matrix, plot_path=plot_path, exp_name=exp_name),
                             df_filenames)

            # Per-experiment grouping of filenames
            df_filenames_dict = dict()
            for df_filename in df_filenames:
                df_filenames_dict.setdefault(df_filename.split('outputs_targets')[0], []).append(df_filename)

            if args.njobs <= 1:
                for key in df_filenames_dict:
                    plot_confusion_matrix(df_filenames_dict[key], plot_path, exp_name, std=True)
            else:
                with Pool(args.njobs) as pool:
                    pool.map(
                        partial(plot_confusion_matrix, plot_path=plot_path, exp_name=exp_name, std=True),
                        list(df_filenames_dict.values()))
