import torch
import learn2learn as l2l
from torch import nn
from torch import optim
from copy import deepcopy
from learn2learn.utils import accuracy

from fsl.approach.tl_module import LightningTLModule


class LightningFreezing(LightningTLModule):
    
    def __init__(self, net, loss=None, **kwargs):
        super().__init__(**kwargs)
        
        # Transfer-Learning specific parameters 
        self.loss = loss or nn.CrossEntropyLoss(reduction="mean")
        self.net = net
        
        self.save_hyperparameters({
            "lr": self.lr,
            "lr_strat": self.lr_strat,
            "scheduler_patience": self.scheduler_patience,
            "t0": self.t0,
            "eta_min": self.eta_min
        })
        
    def on_train_end(self):
        self.net.freeze_backbone()
    
    def pt_step(self, batch, batch_idx):
        data, labels = batch
        labels, le = self.label_encoding(labels)
        logits = self.net(data)
        eval_accuracy = accuracy(logits[0], labels)
        eval_loss = self.loss(logits[0], labels)
        
        return {
            'loss': eval_loss,
            'accuracy': eval_accuracy,
            'labels': labels,
            'logits': logits[0],
            'le_map': le,
        }
        
    def ft_step(self, batch, ft_net):
        x, y = batch

        batch_support, batch_query = l2l.data.utils.partition_task(
            x, y, shots=self.shots)
        
        # Adaptation to the support set
        support, support_labels = batch_support
        support = support.to(self._device)
        support_labels = support_labels.to(self._device)
        support_labels, le = self.label_encoding(support_labels)
        
        ft_net = ft_net or deepcopy(self.net) # Network to fine-tune
        ft_net.train()
        local_opt = optim.Adam(ft_net.parameters(), lr=self.lr)
        local_opt.zero_grad()
        
        support_logits, s_embeddings = ft_net(support, return_features=True)
        adapt_loss = self.loss(support_logits[1], support_labels)
        
        # Calculate gradients and update
        adapt_loss.backward()
        local_opt.step()
        local_opt.zero_grad() # Reset gradients
        
        # Test the fine-tuned network on the query set
        with torch.no_grad():
            ft_net.eval()
            
            query, query_labels = batch_query
            query = query.to(self._device)
            query_labels = query_labels.to(self._device)
            query_labels, _ = self.label_encoding(query_labels)
            
            query_logits, q_embeddings = ft_net(query, return_features=True)
            query_accuracy = accuracy(query_logits[1], query_labels)
            eval_loss = self.loss(query_logits[1], query_labels)
            
        return {
            'loss': eval_loss,
            'accuracy': query_accuracy,
            'query_labels': query_labels,
            'support_labels': support_labels,
            'logits': query_logits[1],
            'le_map': le,
            'support': s_embeddings,
            'query': q_embeddings
        }, ft_net
        
""" def NN(support, support_ys, query):
    support = np.expand_dims(support.transpose(), 0)
    query = np.expand_dims(query, 2)
    diff = np.multiply(query - support, query - support)
    distance = diff.sum(1)
    min_idx = np.argmin(distance, axis=1)
    pred = support_ys[min_idx]
    return pred """
