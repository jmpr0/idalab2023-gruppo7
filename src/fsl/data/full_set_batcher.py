import torch
import random

class FullSetBatchSampler(object):

    def __init__(self, y:torch.Tensor, k_support:int, seed:int):
        """
        This sampler returns one index for a query example and k indices 
        for each class in y, forming the support set. This sampling is performed 
        as many times as the number of elements in y.
        """
        super().__init__()
        self.y = y.tolist()
        self.k_support = k_support
        self.seed = seed

        unique_y = list(set(self.y))
        self.y_to_index = dict(zip(unique_y, [[] for i in range(len(unique_y))]))
        for index,label in enumerate(self.y):
            self.y_to_index[label].append(index)
        
        for indices in self.y_to_index.values():
            assert (len(indices) >= self.k_support
            ), 'full_test_shots > number of samples in a class'

        self.index_to_y = dict(zip([i for i,_ in enumerate(self.y)], self.y))
        keys =  list(self.index_to_y.keys())
        random.Random(self.seed).shuffle(keys)
        self.index_to_y = dict([(key, self.index_to_y[key]) for key in keys])
        

    def __iter__(self):
        for index, query_label in self.index_to_y.items(): # One batch for each elem in y
            meta_batch_indices = []

            meta_batch_indices.append(index) # Picking index of query sample
            self.y_to_index[query_label].remove(index) # Remove query index, so that it'll not be sampled in support set
            for label in self.y_to_index: # Picking k indices per class for support set

                rnd_picks = random.Random(index).sample(self.y_to_index[label], self.k_support)
                meta_batch_indices.extend(rnd_picks)
            
            self.y_to_index[query_label].append(index) # Add query index
            assert len(meta_batch_indices) == len(set(meta_batch_indices)), ('Bad sampling')
            yield meta_batch_indices


    def __len__(self):
        return len(self.y)