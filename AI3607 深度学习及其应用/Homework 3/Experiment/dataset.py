import os
import gzip
import numpy as np
from jittor.dataset.dataset import Dataset


def load_data(path, group='train'):
    image, label = [], []
    if group == 'train':
        with gzip.open(os.path.join(path, 'train-images-idx3-ubyte.gz'), 'rb') as file:
            image = np.frombuffer(file.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28)
        with gzip.open(os.path.join(path, 'train-labels-idx1-ubyte.gz'), 'rb') as file:
            label = np.frombuffer(file.read(), np.uint8, offset=8)
    elif group == 'test':
        with gzip.open(os.path.join(path, 't10k-images-idx3-ubyte.gz'), 'rb') as file:
            image = np.frombuffer(file.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28)
        with gzip.open(os.path.join(path, 't10k-labels-idx1-ubyte.gz'), 'rb') as file:
            label = np.frombuffer(file.read(), np.uint8, offset=8)
    return image.astype(np.float32) / 255.0, label.astype(np.int8)


class MNIST(Dataset):
    def __init__(self, path, group='train', mode='origin', batch=64):
        super().__init__()
        self.path = os.path.join(path, 'mnist_data')
        self.group = group
        self.batch = batch
        self.image, self.label = load_data(self.path, self.group)
        if group == 'train':
            if mode == 'reduced':
                self._reduce()
            elif mode == 'enhanced':
                self._reduce()
                self._enhance()
        self.set_attrs(batch_size=batch, shuffle=True)

    def __len__(self):
        return self.image.shape[0]
    
    def __getitem__(self, index):
        return self.image[index], self.label[index]
    
    def _reduce(self):
        image, label = [], []
        for value in range(10):
            index = np.where(self.label == value)[0]
            if value < 5:
                remain = index.shape[0] // 10
                index = np.random.choice(index, remain, replace=False)
            image.append(self.image[index])
            label.append(self.label[index])
        self.image = np.concatenate(image, axis=0)
        self.label = np.concatenate(label, axis=0)
    
    def _enhance(self):
        image, label = [], []
        for value in range(10):
            index = np.where(self.label == value)[0]
            _image, _label = self.image[index], self.label[index]
            image.append(_image)
            label.append(_label)
            if value < 5:
                for scale in range(1, 10):
                    image.append(_image + 0.03 * scale * np.random.randn(*_image.shape))
                    label.append(_label)
        self.image = np.concatenate(image, axis=0)
        self.label = np.concatenate(label, axis=0)
