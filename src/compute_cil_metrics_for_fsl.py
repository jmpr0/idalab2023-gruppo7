import argparse
import json
import os
import sys
from functools import partial
from glob import glob
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.metrics import *
from tqdm import tqdm
import scipy as sp

tqdm.pandas()


def per_class_davies_bouldin_score(features, targets):
    features = np.array([[v0 for v0 in v] for v in features])
    return davies_bouldin_score(features, targets)


def silhouette_score(targets, features, distance='euclidean'):
    assert distance in ['euclidean']
    features = np.array([[v0 for v0 in v] for v in features])
    silhouette = silhouette_samples(features, targets, metric=distance)
    per_target_silhouette = []
    target_list = np.unique(targets)
    for target in target_list:
        target_idx = [i for i in range(len(targets)) if targets[i] == target]
        per_target_silhouette.append(
            np.mean([silhouette[i] for i in target_idx]))
    return per_target_silhouette


def incoherence_score(targets, features, distance='euclidean'):
    assert distance in ['euclidean', 'mahalanobis']
    features = np.array([[v0 for v0 in v] for v in features])
    if distance == 'euclidean':
        _distance = sp.spatial.distance.euclidean
    elif distance.startswith('mahalanobis'):
        _distance = sp.spatial.distance.mahalanobis
    target_list = np.unique(targets)
    target_idxs = []
    target_means = []
    if distance == 'mahalanobis':
        target_covs = []
    for target in target_list:
        target_idxs.append(
            [i for i in range(len(targets)) if targets[i] == target])
        target_means.append(np.mean([features[i]
                            for i in target_idxs[-1]], axis=0))
        if distance == 'mahalanobis':
            target_covs.append(sp.linalg.pinv(
                np.cov(np.array([[v for v in features[i]] for i in target_idxs[-1]]).T)))
    per_target_incoherence = []
    for i, target in enumerate(target_list):
        intra_dist = sum([_distance(
            *((features[target_idx], target_means[i]) + ((target_covs[i],) if distance == 'mahalanobis' else ())))
            for target_idx in target_idxs[i]])
        inter_dists = []
        for j in range(len(target_list)):
            if i == j:
                continue
            inter_dists.append(sum([_distance(
                *((features[target_idx], target_means[j]) + ((target_covs[j],) if distance == 'mahalanobis' else ())))
                for target_idx in target_idxs[i]]))
        incoherence = intra_dist / sum(inter_dists) if sum(inter_dists) else 0
        per_target_incoherence.append(incoherence)
    return per_target_incoherence


def forgetting_score(df, base_metric='f1_score'):
    forgetting = []
    per_class_max = None
    for i, row in df.iterrows():
        values = row[base_metric]
        if i == 0:
            per_class_max = values
            forgetting.append([np.nan] * len(values))
        else:
            forgetting.append(
                [max_old - v for max_old, v in zip(per_class_max, values)] +
                [np.nan] * (len(values) - len(per_class_max)))
            per_class_max = np.concatenate(
                ([max([old_v, v]) for old_v, v in zip(per_class_max, values)], values[len(per_class_max):]))

    return forgetting


def get_args_dict(df_filename):
    ts = '-'.join(df_filename.split('/')[-1].split('-')[1:]).split('_')[0]
    ts = ts.replace('.parquet', '')
    try:
        return json.loads(open('/'.join(df_filename.split('/')[:-2] + ['args-%s.txt' % ts])).read())
    except FileNotFoundError as _:
        return json.loads(open('/'.join(df_filename.split('/')[:-1] + ['args-%s.txt' % ts])).read())


def compact_metrics(metric, scores, episode_sizes=None, sample_counter=None):
    if episode_sizes is None:
        episode_sizes = [len(scores)]
    if episode_sizes is not None and metric not in ['top_k_accuracy_score', 'confusion_matrix']:
        prev_ep_size = 0
        new_ret = []
        for episode_size in episode_sizes:
            if metric != 'accuracy_score':
                new_ret.append(
                    np.mean(scores[prev_ep_size:prev_ep_size + episode_size]))
            else:
                # Unrolling the per-class accuracies by using the sample_counter, in order to provide the classical
                # accuracy on per-episode classes.
                new_ret.append(
                    sum([a * s for a, s in zip(
                        scores[prev_ep_size:prev_ep_size + episode_size],
                        sample_counter[prev_ep_size:prev_ep_size + episode_size])]) / sum(
                        sample_counter[prev_ep_size:prev_ep_size + episode_size]))
            prev_ep_size += episode_size
        scores = new_ret

    if metric != 'confusion_matrix':
        return scores
    else:
        return [[v0 for v0 in v] for v in scores]


def compute_metrics(metric, targets, scores, is_taw=False):
    ret = []
    for target, score in zip(targets, scores):
        target_indexes = sorted(set(target))
        if is_taw:
            score = np.array([[s[i] for i in target_indexes] for s in score])
        min_target = min(target_indexes)
        target = np.array([t - min_target for t in target])

        kwargs = dict()
        kwargs['y_true'] = target
        if metric != 'top_k_accuracy_score':
            kwargs['y_pred'] = [np.argmax(s) for s in score]
        else:
            if len(target_indexes) == 2:
                score = np.array([v[0] for v in score])
            kwargs['y_score'] = score
            kwargs['k'] = 2
        if metric in ['f1_score', 'precision_score', 'recall_score']:
            kwargs['average'] = None
            kwargs['zero_division'] = 0
        if metric in ['accuracy_score', 'balanced_accuracy_score']:
            matrix = confusion_matrix(**kwargs)
            sample_counter = matrix.sum(axis=1)
            ret.extend(list(matrix.diagonal() / sample_counter))
        else:
            ret.extend(globals()[metric](**kwargs))

    if metric != 'confusion_matrix':
        return ret
    else:
        return [[v0 for v0 in v] for v in ret]


def return_sample_count(target, score):
    matrix = confusion_matrix(target, [np.argmax(s) for s in score])
    sample_counter = matrix.sum(axis=1)
    return sample_counter


def name_and_check_override(filename, discr, override):
    _filename = filename.replace('.parquet', '%s.parquet' % discr)
    if os.path.exists(_filename) and override is None:
        _override = input(
            'File "%s" exists: override [Y, n]? ' % _filename).lower() != 'n'
    else:
        _override = override
    return _filename, _override


def main(df_filename, override=True, skip_features=False):

    print('Elaborating %s' % df_filename)

    dfs = []
    for df_fn in df_filename:
        try:
            dfs.append(pd.read_parquet(df_fn))
        except:
            print('File %s potentially corrupted.' % df_fn)
    df = pd.concat(dfs, ignore_index=True, copy=False)
    df.reset_index(inplace=True, drop=True)

    df_filename = '/'.join(df_filename[0].split('/')[:-1] + ['-'.join(
        ['_'.join(df_filename[0].split('/')[-1].split('-')[0].split('_')[:4]),  # Removing the eventual episode index
         df_filename[0].split('/')[-1].split('-')[-1]])])

    if isinstance(df['Scores'].iloc[0][0][0][0], (list, np.ndarray)):
        df.loc[:, 'Scores'] = df['Scores'].apply(
            lambda x: [np.concatenate([[v1 for v1 in vo] for vo in v], axis=1) for v in x])

    metrics_list = ['balanced_accuracy_score', 'f1_score']


    for file_discr, metric_name, is_taw in zip(
        ['_per_class_taw_metrics', '_per_class_tag_metrics'],
        ['task-aware', 'task-agnostic'],
        [True, False]
        ):
        print('Computing per-class %s metrics' % metric_name)
        df_per_class_metrics_filename, _override = name_and_check_override(
            df_filename, file_discr, override)
        df_per_class_metrics = None
        if os.path.exists(df_per_class_metrics_filename) and not _override:
            try:
                df_per_class_metrics = pd.read_parquet(
                    df_per_class_metrics_filename)
            except:
                pass
        if df_per_class_metrics is None:  # Generate or Override
            df_per_class_metrics = df.progress_apply(
                lambda x: dict(
                    [(metric, compute_metrics(metric, x['Targets'], x['Scores'], is_taw=is_taw)) for metric in metrics_list]),
                axis=1, result_type='expand')
            for metric in metrics_list:
                if metric == 'confusion_matrix':
                    continue
                df_per_class_metrics['forgetting_%s' % metric] = forgetting_score(
                    df_per_class_metrics, metric)
            print('Computing davies-bouldin')
            df_per_class_metrics['davies_bouldin_score'] = df.progress_apply(
                lambda x: per_class_davies_bouldin_score(np.concatenate(x['Features']), np.concatenate(x['Targets'])), axis=1)
            if not skip_features:
                try:
                    for distance in ['euclidean']:
                        print('Computing %s incoherence' % distance)
                        df_per_class_metrics['incoherence_score_%s' % distance] = df.progress_apply(
                            lambda x: incoherence_score(
                                np.concatenate(x['Targets']), np.concatenate(x['Features']), distance), axis=1)
                    print('Computing silhouette')
                    df_per_class_metrics['silhouette_score'] = df.progress_apply(
                        lambda x: silhouette_score(np.concatenate(x['Targets']), np.concatenate(x['Features'])), axis=1)
                except:
                    pass
            df_per_class_metrics.to_parquet(df_per_class_metrics_filename)


# TODO: add incoherence (DISTILLER paper) as metric, applied to extracted features
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Incremental Learning Metrics Computer.')

    parser.add_argument('--exp-name', default=None, type=str,
                        help='Experiment name (default=%(default)s)', nargs='*')
    parser.add_argument('--partial-exp-name', action='store_true',
                        help='If set, files will match *exp_name* instead of *exp_name (default=%(default)s)')
    parser.add_argument('--results-path', type=str, default='../results',
                        help='Results path (default=%(default)s)')
    parser.add_argument('--yes', action='store_true',
                        help='Answer YES to all script requests (default=%(default)s)')
    parser.add_argument('--no', action='store_true',
                        help='Answer NO to all script requests (default=%(default)s)')
    parser.add_argument('--njobs', default=1, type=int,
                        help='Number of processes to spawn (default=%(default)s)')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode (default=%(default)s)')
    parser.add_argument('--file-name', default=None, type=str,
                        help='Model outputs and targets file name (default=%(default)s)', nargs='*')
    parser.add_argument('--nc-first-task', default=None, type=int, nargs='+',
                        help='Number of base classes (default=%(default)s)')
    parser.add_argument('--nc-incr-tasks', default=None, type=int, nargs='+', action='append',
                        help=('Number of classes for each increment, to define for each value passed to '
                              '--nc-first-task (default=%(default)s)'))
    parser.add_argument('--skip-features', action='store_true',
                        help='Skip the features-based metrics (default=%(default)s)')
    parser.add_argument('--is-gate', action='store_true',
                        help='Consider the model as a gate (default=%(default)s)')

    args, _ = parser.parse_known_args(sys.argv)

    assert not (args.yes and args.no), 'YES and NO cannot be set together'
    assert args.exp_name != args.file_name, '--exp-name and --file-name should be considered separately'

    override = args.yes or not args.no if (args.yes or args.no) else None

    if args.njobs > 1 and override is None:
        override = False

    d = '*' if args.partial_exp_name else ''

    if args.file_name is None:
        df_filenames = []
        for exp_name in args.exp_name:
            fns = glob('%s/**/outputs_targets_*.parquet' %
                       (args.results_path), recursive=True)
            print('%s/**/outputs_targets_*.parquet' % (args.results_path))
            df_filenames.extend(
                [fn for fn in fns if 'metrics' not in fn and 'tsne' not in fn and 'backup' not in fn])
    else:
        df_filenames = args.file_name

    # Group filenames by exp_dir and timestamp:
    # experiments falling in the same exp_dir MUST have a different timestamp
    df_filenames_dict = dict()
    for df_filename in df_filenames:
        key = (df_filename.split('/')[-3], df_filename.split('-')[1])
        df_filenames_dict.setdefault(key, []).append(df_filename)

    # Sort filenames by episode
    for key in df_filenames_dict:
        episodes = [
            int(df_filename.split('/')[-1].split('-')[0].split('_')[-1]) for df_filename in df_filenames_dict[key]]
        sorting_index = np.argsort(episodes)
        df_filenames_dict[key] = [df_filenames_dict[key][i]
                                  for i in sorting_index]

    if args.njobs <= 1:
        for df_filename in df_filenames_dict.values():
            main(df_filename, args.skip_features)
    else:
        with Pool(args.njobs) as pool:
            # N.B. override is pushed to the default value True
            pool.map(partial(main, override=override, skip_features=args.skip_features),
                     df_filenames_dict.values())
