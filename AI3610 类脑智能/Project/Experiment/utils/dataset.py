import os

from torch.utils.data import Subset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture


def create_transform(size, train=True):
    transform = [transforms.Resize(size), transforms.RandomCrop(size, padding=size // 8)]
    if train:
        transform += [transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.2, 0.2, 0.2, 0.01)]
    transform += [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]
    return transforms.Compose(transform)


def split_dataset(dataset, train_ratio=0.9, num_classes=10):
    index_label = [[] for _ in range(num_classes)]
    for index, data in enumerate(dataset):
        _, label = data
        index_label[label].append(index)
    index_train, index_test = [], []
    for label in range(num_classes):
        train_count = int(len(index_label[label]) * train_ratio)
        index_train += index_label[label][:train_count]
        index_test += index_label[label][train_count:]
    return Subset(dataset, index_train), Subset(dataset, index_test)


def create_dataset(config):
    if config.name == 'cifar-10' or config.name == 'cifar-10-finetune':
        transform_train = create_transform(config.image_size, train=True)
        transform_test = create_transform(config.image_size, train=False)
        dataset_train = CIFAR10(config.path, train=True, transform=transform_train, download=True)
        dataset_test = CIFAR10(config.path, train=False, transform=transform_test, download=True)
    elif config.name == 'cifar-100' or config.name == 'cifar-100-finetune':
        transform_train = create_transform(config.image_size, train=True)
        transform_test = create_transform(config.image_size, train=False)
        dataset_train = CIFAR100(config.path, train=True, transform=transform_train, download=True)
        dataset_test = CIFAR100(config.path, train=False, transform=transform_test, download=True)
    elif config.name == 'cifar-10-dvs':
        os.makedirs(config.path, exist_ok=True)
        dataset = CIFAR10DVS(config.path, data_type='frame', frames_number=config.time_steps, split_by='number')
        dataset_train, dataset_test = split_dataset(dataset, num_classes=config.num_classes)
    elif config.name == 'dvs-128-gesture':
        os.makedirs(config.path, exist_ok=True)
        dataset_train = DVS128Gesture(config.path, train=True, data_type='frame', frames_number=config.time_steps, split_by='number')
        dataset_test = DVS128Gesture(config.path, train=False, data_type='frame', frames_number=config.time_steps, split_by='number')
    elif config.name == 'imagenet-1k-128':
        transform_train = create_transform(config.image_size, train=True)
        transform_test = create_transform(config.image_size, train=False)
        dataset_train = ImageFolder(os.path.join(config.path, 'train'), transform=transform_train)
        dataset_test = ImageFolder(os.path.join(config.path, 'val'), transform=transform_test)
    loader_train = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    loader_test = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
    return dataset_train, dataset_test, loader_train, loader_test
