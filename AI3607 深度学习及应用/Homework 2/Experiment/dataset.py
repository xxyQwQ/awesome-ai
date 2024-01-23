import os
import pickle
import numpy as np
from jittor.dataset.dataset import Dataset


def load_data(path, group='train'):
    image, label = [], []
    if group == 'train':
        for i in range(1, 6):
            filename = os.path.join(path, 'data_batch_{}'.format(i))
            with open(filename, 'rb') as file:
                data = pickle.load(file, encoding='bytes')
            image.append(np.array(data[b'data'], dtype=np.float32) / 255.0)
            label.append(np.array(data[b'labels'], dtype=np.int32))
        image = np.concatenate(image, axis=0)
        label = np.concatenate(label, axis=0)
    elif group == 'test':
        filename = os.path.join(path, 'test_batch')
        with open(filename, 'rb') as file:
            data = pickle.load(file, encoding='bytes')
        image = np.array(data[b'data'], dtype=np.float32) / 255.0
        label = np.array(data[b'labels'], dtype=np.int32)
    return image.reshape(-1, 3, 32, 32), label


class CIFAR10(Dataset):
    def __init__(self, path, group='train', mode='origin', batch=64):
        super().__init__()
        self.path = os.path.join(path, 'cifar-10-batches-py')
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
