import torch
import numpy as np
import pandas as pd
from h5py import File
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class TrainingDataset(Dataset):
    def __init__(self, audio_file, label_file, label_to_idx, transform=None):
        super(TrainingDataset, self).__init__()
        self.cache = {}
        self.aid_to_h5 = load_dict_from_csv(audio_file, ('audio_id', 'hdf5_path'))
        self.aid_to_label = load_dict_from_csv(label_file, ('filename', 'event_labels'))
        self.aids = list(self.aid_to_label.keys())
        first_aid = self.aids[0]
        with File(self.aid_to_h5[first_aid], 'r') as store:
            self.datadim = store[first_aid].shape[-1]
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.aids)

    def __getitem__(self, index):
        aid = self.aids[index]
        h5_file = self.aid_to_h5[aid]
        if h5_file not in self.cache:
            self.cache[h5_file] = File(h5_file, 'r', libver='latest')
        feat = self.cache[h5_file][aid][()]
        feat = torch.as_tensor(feat).float()
        if self.transform:
            feat = self.transform(feat)
        label = self.aid_to_label[aid]
        target = torch.zeros(len(self.label_to_idx))
        for l in label.split(','):
            target[self.label_to_idx[l]] = 1
        return aid, feat, target


class InferenceDataset(Dataset):
    def __init__(self, audio_file):
        super(InferenceDataset, self).__init__()
        self.cache = {}
        self.aid_to_h5 = load_dict_from_csv(audio_file, ('audio_id', 'hdf5_path'))
        self.aids = list(self.aid_to_h5.keys())
        first_aid = self.aids[0]
        with File(self.aid_to_h5[first_aid], 'r') as store:
            self.datadim = store[first_aid].shape[-1]

    def __len__(self):
        return len(self.aids)

    def __getitem__(self, index):
        aid = self.aids[index]
        h5_file = self.aid_to_h5[aid]
        if h5_file not in self.cache:
            self.cache[h5_file] = File(h5_file, 'r', libver='latest')
        feat = self.cache[h5_file][aid][()]
        feat = torch.as_tensor(feat).float()
        return aid, feat


def load_dict_from_csv(file, cols, sep='\t'):
    if isinstance(file, str):
        df = pd.read_csv(file, sep=sep)
    elif isinstance(file, pd.DataFrame):
        df = file
    output = dict(zip(df[cols[0]], df[cols[1]]))
    return output


def pad(tensorlist, batch_first=True, padding_value=0.):
    if len(tensorlist[0].shape) == 3:
        tensorlist = [ten.squeeze() for ten in tensorlist]
    if isinstance(tensorlist[0], np.ndarray):
        tensorlist = [torch.as_tensor(arr) for arr in tensorlist]
    padded_seq = pad_sequence(tensorlist, batch_first=batch_first, padding_value=padding_value)
    length = [tensor.shape[0] for tensor in tensorlist]
    return padded_seq, length


def sequential_collate(return_length=True, length_idxs=[]):
    def wrapper(batches):
        seqs, lens = [], []
        for idx, data_seq in enumerate(zip(*batches)):
            if isinstance(data_seq[0], (torch.Tensor, np.ndarray)):
                data_seq, data_len = pad(data_seq)
                if idx in length_idxs:
                    lens.append(data_len)
            else:
                data_seq = np.array(data_seq)
            seqs.append(data_seq)
        if return_length:
            seqs.extend(lens)
        return seqs
    return wrapper
