import torch
import numpy as np
from torch import nn


class DaviesBouldinLoss(nn.Module):
    """ 
    Implementation in Pytorch of 'davies_bouldin_score' from sklearn
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, features, labels): 
        EPS = 1.0e-04
        device, dtype = features.device, features.dtype
        n_samples, _ = features.shape
        n_labels = len(torch.unique(labels))
        pdist = nn.PairwiseDistance(p=2)
        
        # Check number of labels
        if not 1 < n_labels < n_samples:
            raise ValueError(
                f'Number of labels is {n_labels}.Valid values are 2 to n_samples - 1 (inclusive)'
            )

        intra_dists = torch.zeros(n_labels, device=device, dtype=dtype)
        centroids = torch.zeros((n_labels, len(features[0])), device=device, dtype=dtype)
        
        for k in range(n_labels):
            k_labels = labels == k
            cluster_k = features[k_labels]
            centroid = cluster_k.mean(axis=0)
            centroids[k] = centroid
            intra_dists[k] = torch.mean(pdist(cluster_k, centroid))

        centroid_distances = torch.zeros(n_labels, n_labels, device=device, dtype=dtype)
        for i, centroid in enumerate(centroids):
            centroid_distance = pdist(centroids, centroid)
            centroid_distances[i] = centroid_distance

        if torch.allclose(intra_dists, torch.zeros(1)) or torch.allclose(centroid_distances, torch.zeros(1)):
            return torch.zeros(1, device=device, dtype=dtype)

        centroid_distances[centroid_distances < EPS] = torch.tensor(np.Inf, device=device, dtype=dtype)
        combined_intra_dists = intra_dists[:, None] + intra_dists
        scores = torch.max(combined_intra_dists / centroid_distances, dim=1)
        return torch.mean(scores.values)
    
    
class SilhouetteLoss(nn.Module):
    """ 
    Implementation in Pytorch of silhouette loss (adapted from pytorch-adapt)
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, features, labels):
        device, dtype = features.device, features.dtype
        unique_labels = torch.unique(labels)
        n_labels = len(unique_labels)
        n_samples = len(features)
        
        if not (1 < n_labels < n_samples):
            raise ValueError(
                f'Number of labels is {n_labels}.Valid values are 2 to n_samples - 1 (inclusive)'
            )
        
        scores = []
        for unique_label in unique_labels:
            curr_cluster = features[labels == unique_label]
            num_elements = len(curr_cluster)
            if num_elements > 1:
                intra_cluster_dists = torch.cdist(curr_cluster, curr_cluster)
                mean_intra_dists = torch.sum(intra_cluster_dists, dim=1) / (
                    num_elements - 1
                )  # minus 1 to exclude self distance
                dists_to_other_clusters = []
                for other_label in unique_labels:
                    if other_label != unique_label:
                        other_cluster = features[labels == other_label]
                        inter_cluster_dists = torch.cdist(curr_cluster, other_cluster)
                        mean_inter_dists = torch.sum(
                            inter_cluster_dists, dim=1) / (len(other_cluster))
                        dists_to_other_clusters.append(mean_inter_dists)
                dists_to_other_clusters = torch.stack(dists_to_other_clusters, dim=1)
                min_dists, _ = torch.min(dists_to_other_clusters, dim=1)
                curr_scores = (min_dists - mean_intra_dists) / (
                    torch.maximum(min_dists, mean_intra_dists)
                )
            else:
                curr_scores = torch.tensor([0], device=device, dtype=dtype)

            scores.append(curr_scores)

        scores = torch.cat(scores, dim=0)
        if len(scores) != n_samples:
            raise ValueError(
                f'scores (shape {scores.shape}) should have same length as feats (shape {features.shape})'
            )
        silhouette_score = torch.mean(scores)
        silhouette_loss = -silhouette_score + 1
        return silhouette_loss