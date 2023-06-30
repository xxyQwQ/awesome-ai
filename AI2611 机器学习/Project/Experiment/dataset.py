import os
import gzip
import pickle
import numpy as np


def MNIST(path, group='train'):
    if group == 'train':
        with gzip.open(os.path.join(path, 'train-images-idx3-ubyte.gz'), 'rb') as file:
            image = np.frombuffer(file.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28) / 255.0
        with gzip.open(os.path.join(path, 'train-labels-idx1-ubyte.gz'), 'rb') as file:
            label = np.frombuffer(file.read(), np.uint8, offset=8)
    elif group == 'test':
        with gzip.open(os.path.join(path, 't10k-images-idx3-ubyte.gz'), 'rb') as file:
            image = np.frombuffer(file.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28) / 255.0
        with gzip.open(os.path.join(path, 't10k-labels-idx1-ubyte.gz'), 'rb') as file:
            label = np.frombuffer(file.read(), np.uint8, offset=8)
    remain = 500 if group == 'train' else 100
    image_list, label_list = [], []
    for value in range(10):
        index = np.where(label == value)[0][:remain]
        image_list.append(image[index])
        label_list.append(label[index])
    image, label = np.concatenate(image_list), np.concatenate(label_list)
    index = np.random.permutation(len(label))
    return image[index], label[index]


def CIFAR10(path, group='train'):
    if group == 'train':
        image_list, label_list = [], []
        for i in range(1, 6):
            filename = os.path.join(path, 'data_batch_{}'.format(i))
            with open(filename, 'rb') as file:
                data = pickle.load(file, encoding='bytes')
            image_list.append(np.array(data[b'data'], dtype=np.float32).reshape(-1, 3, 32, 32) / 255.0)
            label_list.append(np.array(data[b'labels'], dtype=np.int32))
        image, label = np.concatenate(image_list), np.concatenate(label_list)
    elif group == 'test':
        filename = os.path.join(path, 'test_batch')
        with open(filename, 'rb') as file:
            data = pickle.load(file, encoding='bytes')
        image = np.array(data[b'data'], dtype=np.float32).reshape(-1, 3, 32, 32) / 255.0
        label = np.array(data[b'labels'], dtype=np.int32)
    remain = 500 if group == 'train' else 100
    image_list, label_list = [], []
    for value in range(10):
        index = np.where(label == value)[0][:remain]
        image_list.append(image[index])
        label_list.append(label[index])
    image, label = np.concatenate(image_list), np.concatenate(label_list)
    index = np.random.permutation(len(label))
    return image[index], label[index]
