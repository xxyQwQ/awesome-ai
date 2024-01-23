import os

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import LambdaLR
from einops.layers.torch import Reduce
from spikingjelly.activation_based import layer, neuron

from model.transformer import SpikingTransformer


class CosineAnnealing(object):
    def __init__(self, config):
        self.lower_bound = config.lower_bound
        self.cycle_total = config.cycle_total
        self.cycle_warmup = config.cycle_warmup
        self.cycle_decay = config.cycle_decay
        self.discount_factor = 1

    def __call__(self, epoch):
        epoch %= self.cycle_total
        if epoch <= self.cycle_warmup:
            result = self.discount_factor * epoch / self.cycle_warmup
        else:
            result = self.discount_factor * 0.5 * (1 + np.cos((epoch - self.cycle_warmup) / (self.cycle_total - self.cycle_warmup) * np.pi))
        result = max(result, self.lower_bound)
        if epoch == 0:
            self.discount_factor *= self.cycle_decay
        return result


def create_scheduler(optimizer, config):
    function = CosineAnnealing(config)
    scheduler = LambdaLR(optimizer, function)
    return scheduler


def create_model(device, config):
    model = SpikingTransformer(in_channels=config.in_channels, image_size=config.image_size, num_classes=config.num_classes, num_layers=config.num_layers, num_heads=config.num_heads, num_channels=config.num_channels).to(device)
    return model


def inherit_model(device, config):
    model = SpikingTransformer(in_channels=config.in_channels, image_size=config.image_size, num_classes=config.old_classes, num_layers=config.num_layers, num_heads=config.num_heads, num_channels=config.num_channels).to(device)
    model.load_state_dict(torch.load(config.model, map_location=device), strict=False)
    model.classifier_head[2] = layer.Linear(config.num_channels, config.new_classes).to(device)
    return model


def reshape_image(image, repeat=None):
    if len(image.shape) == 4:                                       # static image: (B, C, H, W)
        return image.unsqueeze(0).repeat_interleave(repeat, dim=0)  # (T, B, C, H, W)
    elif len(image.shape) == 5:                                     # neuromorphic image: (B, T, C, H, W)
        return image.transpose(0, 1)                                # (T, B, C, H, W)
    raise ValueError('invalid {}'.format(image.shape))


def visualize_record(path, record):
    plt.figure(figsize=(8, 6))
    plt.plot(record['epoch'], record['loss_train'], label='train')
    plt.plot(record['epoch'], record['loss_test'], label='test')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('loss curve')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(path, 'loss.png'))
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(record['epoch'], record['accuracy_train'], label='train')
    plt.plot(record['epoch'], record['accuracy_test'], label='test')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('accuracy curve')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(path, 'accuracy.png'))
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(record['epoch'], record['time_train'], label='train')
    plt.plot(record['epoch'], record['time_test'], label='test')
    plt.xlabel('epoch')
    plt.ylabel('time')
    plt.title('time curve')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(path, 'time.png'))
    plt.close()


def summarize_record(path, record):
    with open(os.path.join(path, 'summary.txt'), 'w') as file:
        best_epoch = np.argmax(record['accuracy_test'])
        file.write('peak accuracy is achieved at epoch {}\n'.format(best_epoch))
        file.write('corresponding training accuracy is {:.2%}\n'.format(record['accuracy_train'][best_epoch]))
        file.write('corresponding test accuracy is {:.2%}\n'.format(record['accuracy_test'][best_epoch]))

        time_total = np.sum(record['time_train']) + np.sum(record['time_test'])
        file.write('finish {} epochs in {:.2f} seconds\n'.format(len(record['epoch']), time_total))
        file.write('in average {:.2f} seconds per epoch\n'.format(time_total / len(record['epoch'])))
        file.write('where {:.2f} seconds for training and {:.2f} seconds for test\n'.format(np.mean(record['time_train']), np.mean(record['time_test'])))
