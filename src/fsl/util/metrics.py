from glob import glob
from torchmetrics import F1Score
from sklearn import metrics
import numpy as np
import torch


def accuracy(preds, targets):
    acc = (preds.argmax(dim=1).long() == targets.long()).sum().float()
    return (acc / preds.size(0)).numpy()

def f1_score(preds, targets):
    f1 = F1Score(num_classes=len(targets.unique()), average='macro', task='multiclass')
    return (f1(preds.argmax(dim=1).long(), targets)).numpy()

def silhouette_score(feature, label):
    return metrics.silhouette_score(feature, label, metric='euclidean')

def davies_bouldin_score(feature, label):
    return metrics.davies_bouldin_score(feature, label)

def calinski_harabasz_score(feature, label):
    return metrics.calinski_harabasz_score(feature, label)

logit_based_metrics = {
    'acc' : accuracy,
    'f1' : f1_score
}

cluster_based_metrics = {
    'sc' : silhouette_score,
    'db' : davies_bouldin_score,
    'ch' : calinski_harabasz_score
}

def compute_logit_based_metrics(preds, targets):
    value = dict() 
    for metric in logit_based_metrics:
        value[metric] = logit_based_metrics[metric](preds, targets)
    return value


def compute_cluster_based_metrics(feature, label):
    value = dict()
    for metric in cluster_based_metrics:
        value[metric] = cluster_based_metrics[metric](feature, label)
    return value


def compute_metrics(path, epoch_index):
    computed_metrics_tmp = dict()
    # Read logits
    try:
        logits = np.load(f'{path}/logits_ep{epoch_index}.npy') # Old version handling
    except FileNotFoundError:
        logits = np.load(f'{path}/logits_ep{epoch_index}.npz')['logits']
        
    # Read labels
    query_labels = np.load(glob(f'{path}/labels_ep{epoch_index}.npz')[0])['query_labels']
    support_labels = np.load(glob(f'{path}/labels_ep{epoch_index}.npz')[0])['support_labels']
    if len(support_labels.shape) == 3:
        support_labels = support_labels.squeeze(axis=2)

    # Read features
    queries = np.load(glob(f'{path}/queries_ep{epoch_index}.npz')[0])['queries']
    supports = np.load(glob(f'{path}/supports_ep{epoch_index}.npz')[0])['supports']
    
    for i, logit in enumerate(logits): # Compute metrics for an epoch - loop through the epoch lenght
        
        # Get logit based metrics for an episode (e.g., acc, f1...)
        logit = torch.tensor(logit)
        query_label = torch.tensor(query_labels[i])
        computed_logit_metrics = compute_logit_based_metrics(logit, query_label)
        for metric in computed_logit_metrics: # Save the episode metrics in a temp array contained in a dict
            computed_metrics_tmp.setdefault(metric, []).append(computed_logit_metrics[metric])
        
        # Get cluster based metrics for an episode (e.g., sc, db...)
        computed_cluster_metrics_q = compute_cluster_based_metrics(queries[i], query_labels[i])
        computed_cluster_metrics_s = compute_cluster_based_metrics(supports[i], support_labels[i])
        for metric in computed_cluster_metrics_q: # Save the episode metrics in a temp array contained in a dict
            computed_metrics_tmp.setdefault(metric, []).append(computed_cluster_metrics_q[metric])
            computed_metrics_tmp.setdefault(metric, []).append(computed_cluster_metrics_s[metric])
            
    all_metrics = computed_metrics_tmp.keys()  
    # Get mean and std metrics for the current epoch  
    computed_metrics = dict(zip(
        all_metrics,
        [
            (
                np.array(computed_metrics_tmp[k]).mean(), np.array(computed_metrics_tmp[k]).std()
            ) for k in all_metrics
        ]
    ))
    return computed_metrics