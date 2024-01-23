import os
import pickle
import numpy as np
from jittor.dataset.dataset import Dataset


def load_data(path, group='train'):
    image_list = []
    if group == 'train':
        for i in range(1, 6):
            filename = os.path.join(path, 'data_batch_{}'.format(i))
            with open(filename, 'rb') as file:
                data = pickle.load(file, encoding='bytes')
            image_list.append(np.array(data[b'data'], dtype=np.float32) / 255.0)
    elif group == 'test':
        filename = os.path.join(path, 'test_batch')
        with open(filename, 'rb') as file:
            data = pickle.load(file, encoding='bytes')
        image_list.append(np.array(data[b'data'], dtype=np.float32) / 255.0)
    return np.concatenate(image_list, axis=0).reshape(-1, 3, 32, 32)


class PuzzleDataset(Dataset):
    def __init__(self, path, group='train', batch=64, segment=2):
        super().__init__()
        self.path = os.path.join(path, 'cifar-10-batches-py')
        self.group, self.batch, self.segment = group, batch, segment
        self.image = load_data(self.path, self.group)
        self.set_attrs(batch_size=self.batch, shuffle=True)

    def __len__(self):
        return self.image.shape[0]
    
    def __getitem__(self, index):
        raw, cut = self.image[index], []
        height, width = raw.shape[1] // self.segment, raw.shape[2] // self.segment
        for i in range(self.segment):
            for j in range(self.segment):
                cut.append(raw[:, i*height: (i+1)*height, j*width: (j+1)*width])
        image = np.stack(cut, axis=0)
        label = np.random.permutation(self.segment**2)
        return image[label], label
