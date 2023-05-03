import sys
import os
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm

fsl_dir = sys.argv[1]

exp_ckpts = glob('%s/**/checkpoints/*.ckpt' % fsl_dir, recursive=True)

for exp_ckpt in tqdm(exp_ckpts):
    exp_dir = '/'.join(exp_ckpt.split('/')[:-2])
    log_lab_dir = '%s/test_data/' % exp_dir

    epoch = dict()
    epoch['best'] = int(exp_ckpt.split('=')[1].split('.')[0])
    epoch['last'] = max(
        [int(v.split('/')[-1].split('.')[0].split('ep')[-1]) for v in glob('%s/le_ep*.json' % log_lab_dir)])

    for k in epoch:
        logits = np.load(glob('%s/logits_ep%s.np*' % (log_lab_dir, epoch[k]))[0])
        labels = np.load(glob('%s/labels_ep%s.np*' % (log_lab_dir, epoch[k]))[0])
        _logits = [[[[l for l in ls] for ls in np.concatenate(logits, axis=0)]]]
        try:
            _labels = [[[l for l in np.concatenate(labels, axis=0)]]]
        except:
            _labels = [[[l for l in np.concatenate(labels['query_labels'], axis=0)]]]

        existing_results = [fn for fn in glob('%s/outputs_targets_features_%s_*.parquet' % (exp_dir, k))]
        for existing_result in existing_results:
            os.remove(existing_result)

        cil_df = pd.DataFrame({'Scores': _logits, 'Targets': _labels, 'Features': np.nan})
        cil_df.to_parquet('%s/outputs_targets_features_%s_%s.parquet' % (exp_dir, k, epoch[k]))
