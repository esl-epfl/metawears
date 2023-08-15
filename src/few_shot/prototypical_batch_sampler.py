# coding=utf-8
import numpy as np
import torch


class PrototypicalBatchSampler(object):
    """
    PrototypicalBatchSampler: yield a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples',
    In fact at every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
    for 'classes_per_it' random classes.

    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    """

    def __init__(self, labels, classes_per_it, num_samples, iterations):
        """
        Initialize the PrototypicalBatchSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset
        samples indexes will be infered from this iterable.
        - classes_per_it: number of random classes for each iteration
        - num_samples: number of samples for each iteration for each class (support + query)
        - iterations: number of iterations (episodes) per epoch
        """
        super(PrototypicalBatchSampler, self).__init__()
        self.labels = labels
        self.classes_per_it = classes_per_it
        self.sample_per_class = num_samples
        self.iterations = iterations

        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        self.classes = torch.LongTensor(self.classes)

        # create a matrix, indexes, of dim: classes X max(elements per class)
        # fill it with nans
        # for every class c, fill the relative row with the indices samples belonging to c
        # in numel_per_class we store the number of samples for each class/row
        self.idxs = range(len(self.labels))
        seizure_idx_np = np.where(self.labels == 1)[0]
        non_seizure_idx_np = np.where(self.labels == 0)[0]
        self.seizure_idx = torch.LongTensor(seizure_idx_np)
        self.non_seizure_idx = torch.LongTensor(non_seizure_idx_np)

        self.num_seizure = len(seizure_idx_np)
        self.num_non_seizure = len(non_seizure_idx_np)

    def __iter__(self):
        """
        yield a batch of indexes
        """
        spc = self.sample_per_class

        for it in range(self.iterations):
            batch_size = spc * 2
            batch = torch.LongTensor(batch_size)
            sample_idxs = torch.randperm(self.num_seizure)[:spc]
            batch[:spc] = self.seizure_idx[sample_idxs]
            sample_idxs = torch.randperm(self.num_non_seizure)[:spc]
            batch[spc:] = self.non_seizure_idx[sample_idxs]
            # batch = batch[torch.randperm(len(batch))]
            yield batch

    def __len__(self):
        """
        returns the number of iterations (episodes) per epoch
        """
        return self.iterations
