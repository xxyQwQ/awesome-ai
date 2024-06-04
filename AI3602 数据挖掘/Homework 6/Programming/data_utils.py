import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional


class LinkPredictionDataset(Dataset):
    """A torch `Dataset` for link prediction tasks.

    Args:
        data (List[Tuple[int, int]]): A list of (src, dst) tuples.
        labels (List[int], optional): A list of labels.
            Defaults to `None`, which means no provided labels.
            If is `None`, the labels will be set to -1.
    """

    def __init__(self, data: List[Tuple[int, int]], labels: Optional[List[int]] = None):
        self.data = data
        if labels is None:
            labels = [-1] * len(data)
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        src, dst = self.data[idx]
        label = self.labels[idx]
        return src, dst, label


class LinkPredictionCollator:
    """A collator for link prediction tasks.

    This collator should be used with `LinkPredictionDataset` and torch `DataLoader`.
    It collates a batch into three Tensors: srcs, dsts, and labels.
    """

    def __call__(self, batch):
        srcs, dsts, labels = zip(*batch)
        return torch.LongTensor(srcs), torch.LongTensor(dsts), torch.FloatTensor(labels)
