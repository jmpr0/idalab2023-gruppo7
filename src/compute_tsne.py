import argparse
import multiprocessing
import os
import sys
from copy import copy
from functools import partial
from glob import glob
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE as tsne
from tqdm import tqdm

tqdm.pandas()


def main(df_filename, override=True):
    print('Elaborating %s' % df_filename)
    df = pd.read_parquet(df_filename)
    total_repr, total_tar = [], []
    for i, elem in enumerate(df['Features']):
        per_task_repr, per_task_tars, temp, temp_tar = [], [], [], []
        for j, el in enumerate(elem):
            temp.extend([list(i) for i in df['Features'][i][j]])
            temp_tar.extend(df['Targets'][i][j])
            tsamples = tsne(n_components=2, random_state=0, verbose=True, n_jobs=multiprocessing.cpu_count() - 1,
                            learning_rate='auto').fit_transform(np.asarray(temp))
            per_task_repr.append(tsamples.tolist())
            per_task_tars.append(copy(temp_tar))
        total_repr.append(per_task_repr)
        total_tar.append(per_task_tars)
    df = pd.DataFrame({'Representations': total_repr, 'Targets': total_tar})
    df_tsne_filename = df_filename.replace('outputs_targets_features', 'tsne')

    if os.path.exists(df_tsne_filename) and override is None:
        _override = input('File "%s" exists: override [Y, n]? ' % df_tsne_filename).lower() != 'n'
    else:
        _override = override
    if not os.path.exists(df_tsne_filename) or _override:  # Generate or Override
        df.to_parquet(df_tsne_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Incremental Learning TSNE Computer.')

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
        fns = glob('%s/*%s/**/outputs_targets_features*.parquet' % (args.results_path, exp_name), recursive=True)
        df_filenames.extend([fn for fn in fns if 'metrics' not in fn])

    if args.njobs <= 1:
        for df_filename in df_filenames:
            main(df_filename, override)
    else:
        with Pool(args.njobs) as pool:
            # N.B. override is pushed to the default value True
            pool.map(partial(main, override=override), df_filenames)
