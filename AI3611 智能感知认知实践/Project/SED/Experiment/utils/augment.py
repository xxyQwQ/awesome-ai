import torch
import numpy as np
from torchvision.transforms import Compose


class GaussianNoise(object):
    def __init__(self, mean=0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = self.mean + torch.randn_like(tensor) * self.std
        tensor += noise
        return tensor


class TimeShift(object):
    def __init__(self, limit=50):
        self.limit = limit

    def __call__(self, tensor):
        shift = np.random.randint(-self.limit, self.limit)
        tensor = torch.roll(tensor, shift, dims=0)
        return tensor


class TimeMask(object):
    def __init__(self, limit=50):
        self.limit = limit

    def __call__(self, tensor):
        begin = np.random.randint(0, tensor.shape[0])
        length = np.random.randint(0, self.limit)
        end = min(tensor.shape[0], begin + length)
        tensor[begin:end] = 0
        return tensor


class FrequencyShift(object):
    def __init__(self, limit=10):
        self.limit = limit

    def __call__(self, tensor):
        shift = np.random.randint(-self.limit, self.limit)
        tensor = torch.roll(tensor, shift, dims=1)
        return tensor


class FrequencyMask(object):
    def __init__(self, limit=10):
        self.limit = limit

    def __call__(self, tensor):
        begin = np.random.randint(0, tensor.shape[1])
        length = np.random.randint(0, self.limit)
        end = min(tensor.shape[1], begin + length)
        tensor[:, begin:end] = 0
        return tensor


def parse_transform(augmentation):
    transform = []
    if augmentation['gaussian_noise']:
        transform.append(GaussianNoise())
    if augmentation['time_shift']:
        transform.append(TimeShift())
    if augmentation['time_mask']:
        transform.append(TimeMask())
    if augmentation['frequency_shift']:
        transform.append(FrequencyShift())
    if augmentation['frequency_mask']:
        transform.append(FrequencyMask())
    transform = Compose(transform)
    return transform
