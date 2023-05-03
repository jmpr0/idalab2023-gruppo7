import pytorch_lightning as pl


class EpisodicBatcher(pl.LightningDataModule):

    def __init__(
        self,
        train_tasks,
        validation_tasks=None,
        test_tasks=None,
        epoch_length=1,
    ):
        super(EpisodicBatcher, self).__init__()
        self.train_tasks = train_tasks
        if validation_tasks is None:
            print('Validation tasks empty, using train tasks for meta-validation')
            validation_tasks = train_tasks
        self.validation_tasks = validation_tasks
        if test_tasks is None:
            test_tasks = validation_tasks
        self.test_tasks = test_tasks
        for _ in self.test_tasks:  # Needed for fixing the episodes pool
            continue
        self.epoch_length = epoch_length

    @staticmethod
    def epochify(taskset, epoch_length, is_test=False):
        class Epochifier(object):
            i = -1

            def __init__(self, tasks, length):
                self.tasks = tasks
                self.length = length
                self.is_test = is_test

            def __getitem__(self, *args, **kwargs):
                if self.is_test:
                    self.i += 1
                    # Iterate through the episodes pool
                    return self.tasks[self.i]
                else:
                    return self.tasks.sample()  # Draw a random episode

            def __len__(self):
                return self.length

        return Epochifier(taskset, epoch_length)

    def train_dataloader(self):
        return EpisodicBatcher.epochify(
            self.train_tasks,
            self.epoch_length,
        )

    def val_dataloader(self):
        return EpisodicBatcher.epochify(
            self.validation_tasks,
            self.epoch_length,
        )

    def test_dataloader(self):
        length = self.epoch_length
        return EpisodicBatcher.epochify(
            self.test_tasks,
            length,
            True
        )
