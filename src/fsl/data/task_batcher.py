import random

class TaskBatchSampler(object):

    def __init__(
        self, labels_to_indices: dict, w: int, ks: int,
        kq: int, epoch_length: int, seed
    ):
        """
        doc string
        """
        super().__init__()
        self.labels_to_indices = labels_to_indices
        self.labels = self.labels_to_indices.keys()
        assert (
            w <= len(self.labels)
        ), f'Number of ways ({w}) > labels in dataset ({len(self.labels)})' 
        self.w = w
        self.ks = ks
        self.kq = kq
        self.epoch_length = epoch_length
        self.seed = seed
        self.randomizer = -1


    def __iter__(self):
        for _ in range(0, self.epoch_length):
            self.randomizer += 1
            picked_classes = random.Random(
                self.seed + self.randomizer
                ).sample(self.labels, self.w) # Pick W random classes (ways)
            picked_classes.sort()
            picked_samples = []
            
            for picked_class in picked_classes:
                if len(self.labels_to_indices[picked_class]) >= self.kq+self.ks:
                    # Pick Ks + Kq random samples for each selected class
                    picked_samples_for_class = random.Random(
                        self.seed + self.randomizer + picked_class
                    ).sample(self.labels_to_indices[picked_class], self.kq+self.ks)
                    picked_samples.extend(picked_samples_for_class)
                else:
                    # Pick all
                    indices = self.labels_to_indices[picked_class]
                    random.Random(self.seed).shuffle(indices)
                    picked_samples.extend(indices)
            assert (
                len(set(picked_samples)) == len(picked_samples)
            ), 'Bad sampling'
            
            yield picked_samples

    def __len__(self):
        return self.epoch_length