import argparse
import sys
from functools import partial
from glob import glob
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas()
import seaborn as sns


def main(df_filename, override=True):
    print('Elaborating %s' % df_filename)
    df = pd.read_parquet(df_filename)
    all_classes = 40
    base_classes = len(np.unique(df['Targets'][0][0]))
    markers = ['o'] * base_classes
    markers.extend(['X']*(all_classes-base_classes))
    for i, elem in enumerate(df['Representations']):
        for j, el in enumerate(elem):
            # targets_mod = [classorder2[x] for x in df['Targets'][i][j]]
            # targets_mod = df['Targets'][i][j]
            # PLOT
            tsamples = df['Representations'][i][j]
            # number = len(np.unique(targets_mod))
            # classes, counts = np.unique(targets_mod,return_counts=True)
            cmap = plt.get_cmap('gist_rainbow')
            colors = [cmap(i) for i in np.linspace(0, 1, all_classes)]
            for tar, c, m in zip(np.unique(df['Targets'][i][j]), colors, markers):
                index = np.where(np.array(df['Targets'][i][j]) == tar)[0]
                tsamples_tar = [tsamples[x] for x in index]
                sns.scatterplot(x=[item[0] for item in tsamples_tar], y=[item[1] for item in tsamples_tar],
                                label=str(tar), s=8, alpha=0.3, color=c, edgecolor='black', linewidth=0.1, marker=m)
            leg = plt.legend(bbox_to_anchor=(0., 0.99, 1., .35), loc='center', ncol=12, frameon=False,
                             handlelength=1.00, columnspacing=1, fontsize=10, handletextpad=0.20, numpoints=2)
            for lh in leg.legendHandles:
                lh.set_alpha(1)
                lh._sizes = [30]
            plt.xlabel("Task" + str(i) + '_Episode' + str(j))
            plt.tight_layout()
            output_filename = df_filename.replace('results/tsne', 'figures/tsne')
            t_index = df_filename[df_filename.rfind('-') + 1:df_filename.rfind('.')]
            index = df_filename.rfind('/')
            output_filename = '%s/tsne_%s_%s-%s' % (output_filename[:index], i, j, t_index)
            plt.savefig('%s.pdf' % output_filename)
            plt.savefig('%s.png' % output_filename, dpi=300)
            plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Incremental Learning Tsne Plotter.')

    parser.add_argument('--exp-name', default=None, type=str,
                        help='Experiment name (default=%(default)s)', nargs='*')
    parser.add_argument('--results-path', type=str, default='../results',
                        help='Results path (default=%(default)s)')
    parser.add_argument('--yes', action='store_true',
                        help='Answer YES to all script requests (default=%(default)s)')
    parser.add_argument('--no', action='store_true',
                        help='Answer NO to all script requests (default=%(default)s)')
    parser.add_argument('--njobs', default=1, type=int,
                        help='Number of processes to spawn (default=%(default)s)')

    args, _ = parser.parse_known_args(sys.argv)

    # results_path_nas = '/media/nas/datasets/MIRAGE_2020/FSCIL_approaches/hf-project/results'
    # print('WARNING: the results path is forced to on the NAS dir %s.' % results_path_nas)
    # args.results_path = results_path_nas

    assert not (args.yes and args.no), 'YES and NO cannot be set together'

    override = args.yes or not args.no if (args.yes or args.no) else None

    if args.njobs > 1 and override is None:
        override = False

    df_filenames = []
    for exp_name in args.exp_name:
        fns = glob('%s/*%s/**/tsne*.parquet' % (args.results_path, exp_name), recursive=True)
        df_filenames.extend([fn for fn in fns if 'metrics' not in fn])

    if args.njobs <= 1:
        for df_filename in df_filenames:
            main(df_filename, override)
    else:
        with Pool(args.njobs) as pool:
            # N.B. override is pushed to the default value True
            pool.map(partial(main, override=override), df_filenames)
