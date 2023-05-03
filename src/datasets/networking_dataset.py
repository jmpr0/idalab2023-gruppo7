import os
import random
import torch
import numpy as np
import pandas as pd
import socket, struct
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

import util.extract_statistics as extract_stat
import util.transformers as trns


class NetworkingDataset(Dataset):
    """Characterizes a dataset for PyTorch -- this dataset pre-loads all biflows in memory"""

    def __init__(self, data, transform, class_indices=None):
        """Initialization"""
        self.labels = data['y']
        self.images = data['x']
        self.seeds = data['s']
        self.transform = transform
        self._apply_transformations()
        self.class_indices = class_indices

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.images)

    def __getitem__(self, index):
        """Generates one sample of data"""
        x = self.images[index]
        y = self.labels[index]
        return x, y

    def set_modality(self, modality='all'):
        self.modality = modality

    def _apply_transformations(self):
        if self.transform in ['', 'n']:
            if isinstance(self.images, list):
                self._setup_multimodal_input(x_psq=np.array(
                    [x[1] for x in self.images], dtype=np.float32))
            return

        x_trns = []
        y_trns = []
        offset = torch.max(self.labels.unique())
        self.mul = 1
        # If the input is multimodal apply the transformation only to psq input
        xs = np.array([x[1] for x in self.images]) if isinstance(self.images, list) else self.images

        for aug in self.transform:
            print(f'Appying transformation {aug}')
            if aug in ['a', 'd']:
                # Duplicating the samples
                x_trns += [trns.duplicating(x) for x in xs]
                y_trns = self._update_labels_trns(y_trns, offset)
            if aug in ['a', 'j']:
                # Jittering samples' IAT feature
                x_trns += [trns.jittering(x, features=[1], random_state=s)
                           for x, s in zip(xs, self.seeds)]
                y_trns = self._update_labels_trns(y_trns, offset)
            if aug in ['a', 't']:
                # Traslating samples' IAT feature
                x_trns += [trns.traslating(x, features=[1], random_state=s)
                           for x, s in zip(xs, self.seeds)]
                y_trns = self._update_labels_trns(y_trns, offset)
            if aug in ['a', 'c']:
                # Scaling samples' IAT feature
                x_trns += [trns.scaling(x, features=[1]) for x in xs]
                y_trns = self._update_labels_trns(y_trns, offset)
            if aug in ['a', 's']:
                # Slicing the first (.5) and the second (0) half of each sample
                for pr in [.5, 0]:
                    x_trns += [trns.slicing(x, wr=.5, pr=pr) for x in xs]
                    y_trns = self._update_labels_trns(y_trns, offset)
            if aug in ['a', 'h']:
                # Hiding each feature for each sample
                for feature in range(4):
                    x_trns += [trns.features_hiding(x, features=[feature]) for x in xs]
                    y_trns = self._update_labels_trns(y_trns, offset)

        x_trns = np.expand_dims(np.concatenate(x_trns), axis=1)
        y_trns = torch.tensor(y_trns)
        if isinstance(self.images, list):
            # Finalize multimodal transformed input
            self._setup_multimodal_input(x_psq=np.concatenate([xs, x_trns]), mul=self.mul)
        else:
            # Singlemodal input
            self.images = np.concatenate([xs, x_trns])
        self.labels = torch.cat((self.labels, y_trns))

    def _update_labels_trns(self, y_trns, offset):
        y_trns += [torch.add(y, offset*self.mul) for y in self.labels]
        self.mul += 1
        return y_trns  
    
    def _setup_multimodal_input(self, x_psq, mul=1):
        x_pay = np.array([x[0] for x in self.images]*mul, dtype=np.float32)
        x_psq = x_psq.reshape((x_pay.shape[0], -1))
        self.images = np.array([np.concatenate([pay, psq])
                                for pay, psq in zip(x_pay, x_psq)], dtype=np.float32)
              

def get_data(full_path, num_tasks, nc_first_task, nc_incr_tasks, validation, shuffle_classes, class_order=None,
             num_bytes=None, num_pkts=None, fields=None, seed=0, scaling='mms_01', shots=-1, fseed=0, last_class=-1,
             predict_tasks=False, statistical=False, is_ml=False, class_augmentation=False, sample_augmentation=False):
    """Prepare data: dataset splits, task partition, class order"""

    assert num_bytes or num_pkts and len(fields) > 0, (
        'NetworkingDataset requires the definition of num_bytes, or num_pkts and fields, or both.')

    assert scaling in ['mms_01', 'mms_sym1'], (
            'Scaling of features should be one of mms_01 for MinMaxScaler((0,1)), '
            'mms_sym1 for MinMaxScaler((-1,1)), or None to skip. Instead, %s is passed.' % scaling)

    data = {}
    taskcla = []

    path = os.path.join(*full_path.split('/')[:-1])
    dataset_filename = full_path.split('/')[-1]
    dataset_extension = dataset_filename.split('.')[-1]

    mms_range = discr = None
    if scaling == 'mms_01':
        mms_range = (0, 1)
        discr = ''
    elif scaling == 'mms_sym1':
        print('WARNING: inputs are minmax scaled b.t.w. -1 and 1. Padding is maintained at zero.')
        mms_range = (-1, 1)
        discr = '_sym1'

    prep_df_path = os.path.join(path, dataset_filename.replace(
        '.%s' % dataset_extension, '_prep%d%s.%s' % (seed, discr, dataset_extension)))
    all_fields = ['PL', 'IAT', 'DIR', 'WIN', 'LOAD'] #'FLG'
    
    df = None
    
    if statistical:
        # TODO: Preprocessing
        print('Using Statistical Dataset')
        # Statistical feature
        statistical_dataset = dataset_filename.split('.')
        statistical_dataset[0] = statistical_dataset[0] + '_statistical_'+str(num_pkts)
        statistical_dataset = path+'/'+'.'.join(statistical_dataset)

        if not os.path.exists(statistical_dataset):
            print("Calculating statistical dataset...")
            df = pd.read_parquet(full_path)
            df = df[['PL', 'IAT', 'DIR', 'WIN', 'FLG', 'LOAD','LABEL']]
            df = extract_stat.get_statistics(df, num_packets=num_pkts)
            df.to_parquet(statistical_dataset)
        else:
            df = pd.read_parquet(statistical_dataset)

        prep_df_path = os.path.join(statistical_dataset.replace(
            '.%s' % dataset_extension, '_prep%d%s.%s' % (seed, discr, dataset_extension)))
        all_fields = ['PL', 'IAT', 'DIR', 'WIN']
        num_pkts = 17

    valid_idx = None

    if not os.path.exists(prep_df_path):
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.preprocessing import LabelEncoder

        # read parquet
        if df is None:
            df = pd.read_parquet(full_path)

        le = LabelEncoder()
        le.fit(df['LABEL'])
        df['ENC_LABEL'] = le.transform(df['LABEL'])
        np.savetxt(os.path.join(path, 'classes.txt'), le.classes_, fmt='%s')

#         print('WARNING: 0.8 train_ratio is applied.')
#         train_idx, test_idx = train_test_split(df.index, train_size=.8, random_state=seed, stratify=df['ENC_LABEL'])
        print('INFO: 0.7 train_ratio is applied.')
        train_idx, test_idx = train_test_split(df.index, train_size=.7, random_state=seed, stratify=df['ENC_LABEL'])
        
        if not statistical:
            pad = 'FEAT_PAD' in df and 'LOAD_PAD' in df
            for field in all_fields:
                mms = MinMaxScaler(mms_range)
                if pad:
                    pad_field = 'FEAT_PAD' if field != 'LOAD' else 'LOAD_PAD'
                    pad_value = 0.5 if field == 'DIR' else -1
                    df[field] = df[[field, pad_field]].apply(
                        lambda x: np.concatenate((x[field], [pad_value] * x[pad_field])), axis=1)
                mms.fit(np.concatenate(df.loc[train_idx, field].values, axis=0).reshape(-1, 1))
                df['SCALED_%s' % field] = df[field].apply(lambda x: mms.transform(x.reshape(-1, 1)).reshape(-1))
                # TODO: Statistical dataset do not apply minmax scaling
            df = df[['SCALED_%s' % field for field in all_fields] + ['ENC_LABEL']]
        else:
            del df['LABEL']
        
        df['IS_TRAIN'] = df.apply(lambda x: x.name in train_idx, axis=1)
        
        df.to_parquet(prep_df_path)
    else:
        print('WARNING: using pre-processed dataframe.')
#         print('WARNING: 0.8 train_ratio is applied.')
        df = pd.read_parquet(prep_df_path)
        train_idx, test_idx = train_test_split(df.index, train_size=.7, random_state=seed, stratify=df['ENC_LABEL'])
        try:
            check = (np.sort(train_idx) == np.sort(df[df['IS_TRAIN']].index)).all()
        except:
            check = False
        if not check:
            print('WARNING: Indexes does not match. If the dataset is AppClassNet don\'t worry about.')
            train_idx = df[df['IS_TRAIN']].index
            test_idx = df[(~df['IS_TRAIN']) & (~df['IS_VALID'])].index
            valid_idx = df[df['IS_VALID']].index

#     with open('trn_idx_%s_%s_%s_%s.log' % (nc_first_task, seed, fseed, shots), 'w') as fout:
#         fout.write(str([(df.loc[t, 'ENC_LABEL'], t) for t in train_idx if df.loc[t, 'ENC_LABEL'] in class_order[-nc_incr_tasks:]]))
#     exit()

    if is_ml:
        df, fields = extract_stat.flatten(
            df, fields, num_pkts, statistical=statistical, return_fields=True, nofields=['ENC_LABEL', 'IS_TRAIN'])
        df[df.isna()] = -1

    # # Works only with Chen21RNN and PL fields input
    # # TODO: bypass this passage because can introduce approximation errors
    # if not apply_scaling:
    #     print('Restore original PL from scaling')
    #     df['SCALED_PL'] = df['SCALED_PL'].apply(lambda x: [math.floor(v * 1460) for v in x])
    # print(df['SCALED_PL'])
    # input()

    if class_order is None:
        num_classes = len(df['ENC_LABEL'].unique())
        class_order = list(range(num_classes))
    else:
        num_classes = len(class_order)
        class_order = class_order.copy()
    if shuffle_classes:
        np.random.shuffle(class_order)

    if last_class != -1:
        # TODO: for future paper, improve random seed and random shuffle 
        np.random.seed(seed + last_class)
        class_order = [c for c in class_order if c != last_class]
        np.random.shuffle(class_order)
        class_order.append(last_class)

    print('Class order: ', class_order)
    # compute classes per task and num_tasks
    if shots == -1:
        if nc_first_task <= 0:
            cpertask = np.array([num_classes // num_tasks] * num_tasks)
            for i in range(num_classes % num_tasks):
                cpertask[i] += 1
        elif nc_incr_tasks > 0:  # When nc_incr_tasks is set, a fixed-sized set of num_tasks episodes is instantiated
            assert nc_first_task + nc_incr_tasks * (num_tasks - 1) <= num_classes, "tasks want more classes than exist"
            cpertask = np.array([nc_first_task] + [nc_incr_tasks] * (num_tasks - 1))
        else:
            assert nc_first_task <= num_classes, "first task wants more classes than exist"
            remaining_classes = num_classes - nc_first_task
            assert remaining_classes >= (num_tasks - 1), "at least one class is needed per task"  # better minimum 2
            if num_tasks > 1:
                cpertask = np.array([nc_first_task] + [remaining_classes // (num_tasks - 1)] * (num_tasks - 1))
                for i in range(remaining_classes % (num_tasks - 1)):
                    cpertask[i + 1] += 1
            else:
                cpertask = np.array([nc_first_task])
    else:
        cpertask = np.array([nc_first_task] + [nc_incr_tasks] * (num_tasks - 1))
        class_order = (class_order[:nc_first_task].tolist() + 
            [v for i in range(num_tasks - 2) for v in class_order[nc_first_task + nc_first_task * i:nc_first_task + nc_incr_tasks * (i + 1)].tolist()] + 
            class_order[-nc_incr_tasks:].tolist())

    assert num_classes >= cpertask.sum(), "something went wrong, the split does not match num classes"
    cpertask_cumsum = np.cumsum(cpertask)
    init_class = np.concatenate(([0], cpertask_cumsum[:-1]))

    if 'PL_DIR' in fields:
        assert len(fields) == 1, 'Not Implemented Error. Provide a behavior for mixing PL_DIR and other fields.'
        field = fields[0]
        df.loc[:, 'SCALED_%s' % field] = df[['SCALED_PL', 'SCALED_DIR']].apply(
            lambda x: [pl * (dir - .5) * 2 for pl, dir in zip(x['SCALED_PL'], x['SCALED_DIR'])], axis=1)
        # TODO: currently the SCALED_PL_DIR is obtained from pre-scaled version of the two fields.
        # mms = MinMaxScaler(mms_range)
        # pad_field = 'FEAT_PAD'
        # pad_value = 0.
        # df[field] = df[[field, pad_field]].apply(
        #     lambda x: np.concatenate((x[field], [pad_value] * x[pad_field])), axis=1)
        # mms.fit(np.concatenate(df.loc[train_idx, field].values, axis=0).reshape(-1, 1))
        # df['SCALED_%s' % field] = df[field].apply(lambda x: mms.transform(x.reshape(-1, 1)).reshape(-1))
    # print(df['SCALED_PL'])
    # print(df['SCALED_DIR'])
    # print(df['SCALED_PL_DIR'])
    # input()

    # initialize data structure
    for tt in range(num_tasks):
        data[tt] = {}
        data[tt]['name'] = 'task-' + str(tt)
        data[tt]['trn'] = {'x': [], 'y': [], 's': []}
        data[tt]['val'] = {'x': [], 'y': [], 's': []}
        data[tt]['tst'] = {'x': [], 'y': [], 's': []}

    # If the few-shot modality is enabled, the dataset is shuffled basing on fseed
    # N.B. this function is applyied w/o impacting the external flow random state
    if shots > -1:
        st0 = np.random.get_state()
        np.random.seed(fseed)
        np.random.shuffle(np.asarray(train_idx))
        np.random.set_state(st0)

    if is_ml:
        train_zip = zip(df.loc[train_idx, fields].values, df.loc[train_idx, 'ENC_LABEL'], df.loc[train_idx].index)
        test_zip = zip(df.loc[test_idx, fields].values, df.loc[test_idx, 'ENC_LABEL'], df.loc[test_idx].index)
        if valid_idx is not None:
            valid_zip = zip(df.loc[valid_idx, fields].values, df.loc[valid_idx, 'ENC_LABEL'], df.loc[valid_idx].index)
    elif num_bytes and num_pkts:
        train_zip = zip(df.loc[train_idx, ['SCALED_LOAD'] + ['SCALED_%s' % field for field in fields]].values,
                        df.loc[train_idx, 'ENC_LABEL'], df.loc[train_idx].index)
        test_zip = zip(df.loc[test_idx, ['SCALED_LOAD'] + ['SCALED_%s' % field for field in fields]].values,
                       df.loc[test_idx, 'ENC_LABEL'], df.loc[test_idx].index)
        if valid_idx is not None:
            valid_zip = zip(df.loc[valid_idx, ['SCALED_LOAD'] + ['SCALED_%s' % field for field in fields]].values,
                       df.loc[valid_idx, 'ENC_LABEL'], df.loc[valid_idx].index)
    elif num_bytes:
        train_zip = zip(df.loc[train_idx, 'SCALED_LOAD'], df.loc[train_idx, 'ENC_LABEL'], df.loc[train_idx].index)
        test_zip = zip(df.loc[test_idx, 'SCALED_LOAD'], df.loc[test_idx, 'ENC_LABEL'], df.loc[test_idx].index)
        if valid_idx is not None:
            valid_zip = zip(df.loc[valid_idx, 'SCALED_LOAD'], df.loc[valid_idx, 'ENC_LABEL'], df.loc[valid_idx].index)
    else:
        train_zip = zip(df.loc[train_idx, ['SCALED_%s' % field for field in fields]].values,
                        df.loc[train_idx, 'ENC_LABEL'], df.loc[train_idx].index)
        test_zip = zip(df.loc[test_idx, ['SCALED_%s' % field for field in fields]].values,
                       df.loc[test_idx, 'ENC_LABEL'], df.loc[test_idx].index)
        if valid_idx is not None:
            valid_zip = zip(df.loc[valid_idx, ['SCALED_%s' % field for field in fields]].values,
                       df.loc[valid_idx, 'ENC_LABEL'], df.loc[valid_idx].index)
        
    nsamples_per_class = df.loc[train_idx, 'ENC_LABEL'].value_counts().to_dict()
    fs_counting_samples = dict([(v, min(shots, nsamples_per_class[v])) for v in class_order[-nc_incr_tasks:]])
    if shots > -1:
        print('Shots per Class:', fs_counting_samples)
    
    # ALL OR TRAIN
#     with open('trn_data_ep1_%s_%s_%s_%s.log' % (nc_first_task, seed, fseed, shots), 'w') as fout:
    for this_row in train_zip:
        this_biflow, this_label, this_ip = this_row[:-1], this_row[-2], this_row[-1]
        if this_label not in class_order or fs_counting_samples.get(this_label, -1) == 0:
            continue
        if this_label in fs_counting_samples:
#           fout.write(str(this_biflow) + str(this_label) + '\n')
            fs_counting_samples[this_label] -= 1
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)

        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        if this_task >= num_tasks:
            continue

        this_sample = format_data(this_biflow, is_ml, num_bytes, num_pkts, fields)
        this_seed = get_random_state(this_ip)

        data[this_task]['trn']['x'].append(this_sample)
        data[this_task]['trn']['y'].append(this_label - init_class[this_task])
        data[this_task]['trn']['s'].append(this_seed)
#     exit()

    # ALL OR TEST
    for this_row in test_zip:
        this_biflow, this_label, this_ip = this_row[:-1], this_row[-2], this_row[-1]
        this_label = int(this_label)
        
        if this_label not in class_order:
            continue
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)

        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        if this_task >= num_tasks:
            continue

        this_sample = format_data(this_biflow, is_ml, num_bytes, num_pkts, fields)
        this_seed = get_random_state(this_ip)

        data[this_task]['tst']['x'].append(this_sample)
        data[this_task]['tst']['y'].append(this_label - init_class[this_task])
        data[this_task]['tst']['s'].append(this_seed)

    # check classes
    for tt in range(num_tasks):
        data[tt]['ncla'] = len(np.unique(data[tt]['trn']['y']))
        assert data[tt]['ncla'] == cpertask[tt], "something went wrong splitting classes"

    # validation
    if valid_idx is None and validation > 0.0:
        for tt in data.keys():
            for cc in range(data[tt]['ncla']):
                cls_idx = list(np.where(np.asarray(data[tt]['trn']['y']) == cc)[0])
                rnd_img = random.sample(cls_idx, int(np.ceil(len(cls_idx) * validation)))
                rnd_img.sort(reverse=True)
                for ii in range(len(rnd_img)):
                    data[tt]['val']['x'].append(data[tt]['trn']['x'][rnd_img[ii]])
                    data[tt]['val']['y'].append(data[tt]['trn']['y'][rnd_img[ii]])
                    data[tt]['trn']['x'].pop(rnd_img[ii])
                    data[tt]['trn']['y'].pop(rnd_img[ii])
    else:
        # ALL OR VALID
        for this_row in valid_zip:
            this_biflow, this_label, this_ip = this_row[:-1], this_row[-2], this_row[-1]
            this_label = int(this_label)
            
            if this_label not in class_order:
                continue
            # If shuffling is false, it won't change the class number
            this_label = class_order.index(this_label)

            # add it to the corresponding split
            this_task = (this_label >= cpertask_cumsum).sum()
            if this_task >= num_tasks:
                continue

            this_sample = format_data(this_biflow, is_ml, num_bytes, num_pkts, fields)
            this_seed = get_random_state(this_ip)

            data[this_task]['val']['x'].append(this_sample)
            data[this_task]['val']['y'].append(this_label - init_class[this_task])
            data[this_task]['val']['s'].append(this_seed)

    if predict_tasks:
        for tt in data.keys():
            data[tt]['tst']['y'] = [0] * len(data[tt]['tst']['y'])
            data[tt]['val']['y'] = [0] * len(data[tt]['val']['y'])
            data[tt]['trn']['y'] = [0] * len(data[tt]['trn']['y'])
            data[tt]['ncla'] = 1

    # other
    n = 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n
    
    return data, taskcla, class_order


def format_data(this_biflow, is_ml, num_bytes, num_pkts, fields):
    
    if is_ml:
        this_sample = this_biflow[0]
    elif num_bytes and num_pkts:
        this_sample = (
            this_biflow[0][0][:num_bytes].reshape(1, num_bytes).astype('float32'),
            np.array([tb[:num_pkts] for tb in this_biflow[0][1:]]).reshape(1, len(fields), num_pkts).transpose(0, 2, 1).astype('float32')
        )
    elif num_bytes:
        this_sample = this_biflow[0][:num_bytes].reshape(1, num_bytes).astype('float32')
    else:
        this_sample = np.array([tb[:num_pkts] for tb in this_biflow[0]]).reshape(1, len(fields), num_pkts).transpose(0, 2, 1).astype('float32')
    # if 'PL_DIR' in fields:
    #     # TODO: generalize this behavior. This branch places the features as channel of a 2D array.
    #     this_sample = np.squeeze(this_sample, axis=0).transpose(1, 0)

    return this_sample


def get_random_state(idx):
    """
    Compute the random_state from the server IP address 
    """
    server_ip = idx.split(',')[2]
    return struct.unpack("!L", socket.inet_aton(server_ip))[0]


def tl_split(dc, num_bytes=None, num_pkts=None, fields=None, seed=0, scaling='mms_01',
             nc_pretrain=None, pt_only=False):
    # TODO: Augs also for ft
    assert num_bytes or num_pkts and len(fields) > 0, (
        'NetworkingDataset requires the definition of num_bytes, or num_pkts and fields, or both.'
    )
    assert scaling in ['mms_01', 'mms_sym1'], (
        'Scaling of features should be one of mms_01 for MinMaxScaler((0,1)) '
        'or mms_sym1 for MinMaxScaler((-1,1)). Instead, %s is passed.' % scaling
    )

    full_path = dc['path']
    fs_split = dc['fs_split']

    x, y, _ = _get_x_y(full_path, scaling, num_bytes, num_pkts, fields, seed)

    if nc_pretrain is not None:
        classes = np.concatenate([fs_split[f'{k}_classes']
                                 for k in ['train', 'val', 'test']])
        pretrain_classes = torch.tensor(classes[:nc_pretrain])
        finetune_classes = torch.tensor(classes[nc_pretrain:])
    else:
        pretrain_classes = torch.tensor(fs_split['train_classes'])
        finetune_classes = torch.tensor(fs_split['test_classes'])

    x_pt, y_pt =  _dataset_from_labels(x, y, pretrain_classes, return_xy=True) 
    pt_ways = len(pretrain_classes)
    ft_ways = len(finetune_classes)
    
    train_set, val_set, test_set = _hold_out(x_pt, y_pt, seed)
    
    if pt_only:
        return [pt_ways], train_set, test_set, val_set, None
    else:
        finetune_set = _dataset_from_labels(x, y, finetune_classes) 
        
        return [pt_ways, ft_ways], train_set, test_set, val_set, finetune_set


def meta_split(dc, num_bytes=None, num_pkts=None, fields=None, augs='', seed=0,
               scaling='mms_01', nc_train=None, nc_val=None, nc_test=None, hold_out=False):
    assert num_bytes or num_pkts and len(fields) > 0, (
        'NetworkingDataset requires the definition of num_bytes, or num_pkts and fields, or both.'
    )
    assert scaling in ['mms_01', 'mms_sym1'], (
        'Scaling of features should be one of mms_01 for MinMaxScaler((0,1)) '
        'or mms_sym1 for MinMaxScaler((-1,1)). Instead, %s is passed.' % scaling
    )

    full_path = dc['path']
    fs_split = dc['fs_split']

    x, y, indices = _get_x_y(
        full_path, scaling, num_bytes, num_pkts, fields, seed)

    if hold_out:
        sets = _hold_out(x, y, seed) 
    else:
        sets = _class_split(x, y, indices, fs_split, nc_train, nc_val, nc_test, augs) 
    return sets


def _class_split(x, y, indices, fs_split, nc_train, nc_val, nc_test, augs):
    
    if nc_train is not None and nc_val is not None and nc_test is not None:
        classes = np.concatenate([fs_split[f'{k}_classes']
                                 for k in ['train', 'val', 'test']])
        train_classes = torch.tensor(classes[:nc_train])
        val_classes = torch.tensor(classes[-(nc_val + nc_test):-nc_test])
        test_classes = torch.tensor(classes[-nc_test:])
    else:
        train_classes = torch.tensor(fs_split['train_classes'])
        val_classes = torch.tensor(fs_split['val_classes'])
        test_classes = torch.tensor(fs_split['test_classes'])

    train_set = _dataset_from_labels(x, y, train_classes, indices, augs)
    val_set = _dataset_from_labels(x, y, val_classes)
    test_set = _dataset_from_labels(x, y, test_classes)

    return train_set, val_set, test_set


def _hold_out(x, y, seed):
    
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=.7, random_state=seed, stratify=y)
    
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, train_size=.9, random_state=seed, stratify=y_train)
    
    data_train = dict(zip(['x', 'y', 's'], [x_train, y_train, None]))
    data_test = dict(zip(['x', 'y', 's'], [x_test, y_test, None]))
    data_val = dict(zip(['x', 'y', 's'], [x_val, y_val, None]))
    
    train_set = NetworkingDataset(data_train, transform='')
    test_set = NetworkingDataset(data_test, transform='')
    val_set = NetworkingDataset(data_val, transform='')
    
    return train_set, val_set, test_set


def _dataset_from_labels(x, y, class_set, indices=None, augs='', return_xy=False):
    class_mask = (y[:, None] == class_set[None, :]).any(dim=-1)
    if isinstance(x, list):
        x = [e for e, m in zip(x, class_mask) if m]
    else:
        x = x[class_mask]
    y = y[class_mask]
    
    if return_xy:
        return x, y

    indices = ([get_random_state(idx) for idx in indices[class_mask]]
               if indices is not None else [])
    data = dict(zip(['x', 'y', 's'], [x, y, indices]))
    return NetworkingDataset(data, transform=augs)


def _get_x_y(full_path, scaling, num_bytes, num_pkts, fields, seed):

    path = os.path.join(*full_path.split('/')[:-1])
    dataset_filename = full_path.split('/')[-1]
    dataset_extension = dataset_filename.split('.')[-1]

    if scaling == 'mms_01':
        mms_range = (0, 1)
    else:  # scaling == 'mms_sym1':
        print('WARNING: inputs are minmax scaled b.t.w. -1 and 1. Padding is maintained at zero.')
        mms_range = (-1, 1)
    discr = '_sym1' if mms_range == (-1, 1) else ''

    prep_df_path = os.path.join(
        path, dataset_filename.replace(
            '.%s' % dataset_extension,
            '_prep%d%s.%s' % (seed, discr, dataset_extension)
        )
    )
    if not os.path.exists(prep_df_path):
        # First time reading the dataset
        from sklearn.preprocessing import LabelEncoder
        from sklearn.preprocessing import MinMaxScaler

        print('Processing dataframe...')
        # Read parquet
        df = pd.read_parquet(full_path)

        # Label encoding
        le = LabelEncoder()
        le.fit(df['LABEL'])
        df['ENC_LABEL'] = le.transform(df['LABEL'])
        label_conv = dict(zip(le.classes_, le.transform(le.classes_)))
        with open(os.path.join(path, 'classes_map.txt'), 'w') as fp:
            fp.write(str(label_conv))

        # Fields scaling & padding
        all_fields = ['PL', 'IAT', 'DIR', 'WIN', 'FLG', 'LOAD']
        has_pad_col = 'FEAT_PAD' in df and 'LOAD_PAD' in df

        for field in all_fields:
            mms = MinMaxScaler(mms_range)
            if has_pad_col:
                pad_field = 'FEAT_PAD' if field != 'LOAD' else 'LOAD_PAD'
                pad_value = 0.5 if field == 'DIR' else -1
                df[field] = df[[field, pad_field]].apply(
                    lambda x: np.concatenate((x[field], [pad_value] * x[pad_field])), axis=1)
            mms.fit(np.concatenate(df[field].values, axis=0).reshape(-1, 1))
            df['SCALED_%s' % field] = df[field].apply(
                lambda x: mms.transform(x.reshape(-1, 1)).reshape(-1)
            )

        df = df[['SCALED_%s' % field for field in all_fields] + ['ENC_LABEL']]
        df.to_parquet(prep_df_path)
    else:
        print('WARNING: using pre-processed dataframe.')
        df = pd.read_parquet(prep_df_path)  # , engine='fastparquet')

    #  Get x and y
    if num_bytes and num_pkts:
        # Mimetic input
        columns = ['SCALED_%s' % field for field in fields]
        x = []
        for load, row in zip(df['SCALED_LOAD'], df[columns].to_numpy()):
            x.append((
                load[:num_bytes],
                np.expand_dims(np.stack([r[:num_pkts] for r in row]).swapaxes(0, 1), axis=0)))
    elif num_bytes:
        # Wang input
        x = np.array([np.reshape(load[:num_bytes], (1, num_bytes))
            for load in df['SCALED_LOAD']], dtype=np.float32)
    else:
        # Lopez input
        columns = ['SCALED_%s' % field for field in fields]
        x = np.array([np.expand_dims(
            np.stack([r[:num_pkts] for r in row]).swapaxes(0, 1), axis=0) 
                      for row in df[columns].to_numpy()], dtype=np.float32)

    y = [label for label in df['ENC_LABEL']]
    y = torch.tensor(y)
    return x, y, df.index
