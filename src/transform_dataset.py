# coding: utf-8
from util.transformers import *
from functools import partial
import pandas as pd
from tqdm import tqdm


def saturate(x, minimum=0, maximum=np.inf):
    x[x < minimum] = minimum
    x[x > maximum] = maximum
    return x


df = pd.read_parquet(
    '/media/nas/datasets/MIRAGE_2020/modeling_and_prediction/datasets/class_incremental/dataset_100pkt_5f_payload_df_exact_noNullver_no0load_noMaps_b8829cf2.parquet')

seed = 1
num_pkts = 10
padding_dict = dict(PL=-1, DIR=.5, IAT=-1, WIN=-1)
minimum_dict = dict(PL=df['PL'].apply(lambda x: min(x)).min(),
                    DIR=df['DIR'].apply(lambda x: min(x)).min(),
                    IAT=df['IAT'].apply(lambda x: min(x)).min(),
                    WIN=df['WIN'].apply(lambda x: min(x)).min())
maximum_dict = dict(PL=df['PL'].apply(lambda x: max(x)).max(),
                    DIR=df['DIR'].apply(lambda x: max(x)).max(),
                    IAT=df['IAT'].apply(lambda x: max(x)).max(),
                    WIN=df['WIN'].apply(lambda x: max(x)).max())

transformed_dfs = []

for transform, transform_lbl in tqdm([
    (partial(jittering, u=0, s=5), 'JITTERED'),
    (partial(scaling, u=1, s=.2), 'SCALED'),
    (partial(magnitude_warping, u=1, s=.2), 'WARPED'),
    (partial(slicing, wr=.5, pr=0, strategy='constant', fill_value=0), 'SLICED1H'),
    (partial(slicing, wr=.5, pr=.5, strategy='constant', fill_value=0), 'SLICED2H'),
]):
    df_to_transform = df.copy()
    for feat in tqdm(['PL', 'DIR', 'IAT', 'WIN']):

        df_to_transform[feat] = df_to_transform.loc[:, feat].apply(lambda x: x[:num_pkts])

        if feat != 'DIR' or 'SLICE' in transform_lbl:
            np.random.seed(seed)
            df_to_transform[feat] = df_to_transform.loc[:, feat].apply(
                transform
                if 'SLICE' not in transform_lbl or feat != 'DIR'
                else partial(transform, fill_value=-1)).apply(
                partial(saturate, minimum=minimum_dict[feat], maximum=maximum_dict[feat])
                if 'SLICE' not in transform_lbl
                else (lambda x: x)).apply(
                np.round
                if feat in ['PL', 'WIN']
                else (lambda x: x))

        df_to_transform[feat] = df_to_transform.loc[:, feat].apply(
            lambda x: list(x) + [padding_dict[feat]] * (num_pkts - len(x)))

    df_to_transform['LABEL'] = df_to_transform.loc[:, 'LABEL'].apply(lambda x: '%s_%s' % (x, transform_lbl))

    transformed_dfs.append(df_to_transform)

for transformed_df, transform_lbl in zip(transformed_dfs, ['JITTERED', 'SCALED', 'WARPED', 'SLICED1H', 'SLICED2H']):
    transformed_df[['PL', 'DIR', 'IAT', 'WIN', 'LABEL']].to_parquet(
        '/media/nas/datasets/MIRAGE_2020/modeling_and_prediction/datasets/class_incremental/dataset_100pkt_5f_payload_df_exact_noNullver_no0load_noMaps_trans%s_b8829cf2.parquet' % transform_lbl.capitalize())
