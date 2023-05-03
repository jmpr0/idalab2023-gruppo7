# coding: utf-8
from datasets.data_loader import *
from networks.network import *
from networks.lopez17cnn import *
import torch
from datasets.exemplars_dataset import ExemplarsDataset
from MulticoreTSNE import MulticoreTSNE
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import os
from sklearn.metrics import f1_score
import json
from glob import glob
import utils
from time import process_time_ns as process_time

torch.multiprocessing.set_sharing_strategy('file_system')

results_path = '/media/nas/datasets/MIRAGE_2020/FSCIL_approaches/hf-project/aws_results/'
exp_dir = 'mirage_generic_jointft_1k_samples_base39_incr1_stop2/'
fns = glob(results_path + exp_dir + 'models/task0-*.ckpt')

for fn in fns:
    base_model = torch.load(fn, map_location='cpu')
    ts = fn.split('-')[-1].replace('.ckpt', '')
    args = json.loads(open(results_path + exp_dir + 'args-%s.txt' % ts).read())
    stdout = open(results_path + exp_dir + 'stdout-%s.txt' % ts).readlines()
    class_order = eval([v for v in stdout[:10] if 'Class order' in v][0].split(':')[-1])

    utils.seed_everything(seed=args['seed'])

    base_class_order = class_order[:args['nc_first_task']]
    sorting_dict = dict(
        [(k, v) for k, v in enumerate(base_class_order)]
    )

    os.chdir('datasets/')
    trn_load, val_load, tst_load, taskcla = get_loaders(
        datasets=args['datasets'], num_tasks=args['num_tasks'], nc_first_task=args['nc_first_task'],
        nc_incr_tasks=args['nc_incr_tasks'], batch_size=args['batch_size'], num_workers=args['num_workers'],
        pin_memory=args['pin_memory'], validation=args['validation'], num_bytes=args['num_bytes'],
        num_pkts=args['num_pkts'], fields=args['fields'], seed=args['seed'], shots=args['shots'],
        fseed=args['fseed'], last_class=args['last_class'], predict_tasks=args['predict_tasks'],
        statistical=args['statistical'], is_ml=False)
    os.chdir('../')

    model = Lopez17CNN(num_pkts=10, num_fields=4, out_features_size=None)

    net = LLL_Net(model, remove_existing_head=True, modality='all', activate_features=True)
    net.add_head(num_outputs=39)
    print(sum([p for p in net.parameters()][0]).sum())
    net.load_state_dict(base_model)
    print(sum([p for p in net.parameters()][0]).sum())

    true = np.array(tst_load[0].dataset.labels)

    net.eval()
    preds = np.array(
        [p.detach().numpy() for p in net(torch.Tensor(np.array(tst_load[0].dataset.images)))]).argmax(-1)[0]
    print(f1_score(true, preds, average='macro'))

    Appr_ExemplarsDataset = ExemplarsDataset
    first_train_ds = trn_load[0].dataset
    transform, class_indices = first_train_ds.transform, first_train_ds.class_indices

    utils.seed_everything(seed=args['seed'])
    trn = np.array(trn_load[0].dataset.images)
    print('Executing t-SNE')
    t = process_time()
    trn_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3, n_jobs=64,
                        verbose=True).fit_transform(trn.reshape(trn.shape[0], np.prod(trn.shape[1:])))
    print('Elapsed time for t-SNE:', '%.4f' % ((process_time() - t) * 1e-9), 's')

    df_embedded = pd.DataFrame(trn_embedded, columns=['x', 'y'])
    labels = np.array(trn_load[0].dataset.labels)
    df_embedded.loc[:, 'label'] = labels

    sel_loader = torch.utils.data.DataLoader(
        trn_load[0].dataset, batch_size=trn_load[0].batch_size, shuffle=False, num_workers=trn_load[0].num_workers,
        pin_memory=trn_load[0].pin_memory)

    mem_indexs = []
    for sel in ['ensemble', 'random', 'herding', 'entropy', 'distance']:
        exemplars_dataset = Appr_ExemplarsDataset(
        transform, class_indices, num_exemplars=1000, num_exemplars_per_class=0, exemplar_selection=sel,
        is_networking=True, base_class_index=0)

        utils.seed_everything(seed=args['seed'])
        # exemplars_dataset.collect_exemplars(net, trn_load[0], val_load[0].dataset.transform)
        print('Executing %s' % sel.capitalize())
        t = process_time()
        mem_index = exemplars_dataset.exemplars_selector._select_indices(
            net, sel_loader, exemplars_dataset.exemplars_selector._exemplars_per_class_num(net),
            val_load[0].dataset.transform)
        print('Elapsed time for %s:' % sel.capitalize(), '%.4f' % ((process_time() - t) * 1e-9), 's')
        mem_indexs.append(mem_index)

        df_embedded.loc[:, sel] = False
        df_embedded.loc[mem_index, sel] = True

    df_embedded.to_parquet(
        'mem_embedded_%s_%s_%s_%s.parquet' % (args['nc_first_task'], args['seed'], args['last_class'], ts))

    for label in set(labels):  # For each label, choice for the best selector
        label_index = [i for i, l in enumerate(labels) if l == label]
        ensemble_mem_index = [i for i in mem_indexs[0] if i in label_index]
        for mem_index in mem_indexs:
            print(label, len([v for v in ensemble_mem_index if v in mem_index]))
        # input()
