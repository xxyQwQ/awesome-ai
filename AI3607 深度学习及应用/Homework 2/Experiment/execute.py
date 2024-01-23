import sys
import numpy as np
import jittor as jt
from dataset import CIFAR10
from model import VGGModel
import utils


class Logger:
    def __init__(self, path='./output/output.txt'):
        self.terminal = sys.stdout
        self.file = open(path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)

    def flush(self):
        pass


def execute():
    sys.stdout = Logger('./output/output.txt')

    print('<experiment> vgg16 on cifar10')
    model = VGGModel()
    loader_train = CIFAR10('./dataset', group='train', mode='origin', batch=1024)
    loader_test = CIFAR10('./dataset', group='test', mode='origin', batch=1024)
    record_train, record_test = utils.train(model, (loader_train, loader_test), total_epoch=100, learning_rate=1e-3, save_name='vgg_origin')
    utils.plot(record_train, record_test, save_name='vgg_origin')
    print()

    print('<experiment> vgg16 on reduced cifar10')
    model = VGGModel()
    loader_train = CIFAR10('./dataset', group='train', mode='reduced', batch=1024)
    loader_test = CIFAR10('./dataset', group='test', mode='reduced', batch=1024)
    record_train, record_test = utils.train(model, (loader_train, loader_test), total_epoch=100, learning_rate=1e-3, save_name='vgg_reduced')
    utils.plot(record_train, record_test, save_name='vgg_reduced')
    print()

    print('<experiment> vgg16 on enhanced cifar10')
    model = VGGModel()
    loader_train = CIFAR10('./dataset', group='train', mode='enhanced', batch=1024)
    loader_test = CIFAR10('./dataset', group='test', mode='enhanced', batch=1024)
    record_train, record_test = utils.train(model, (loader_train, loader_test), total_epoch=100, learning_rate=1e-3, save_name='vgg_enhanced')
    utils.plot(record_train, record_test, save_name='vgg_enhanced')
    print()

    print('<experiment> generalized vgg16 on reduced cifar10')
    model = VGGModel()
    loader_train = CIFAR10('./dataset', group='train', mode='reduced', batch=1024)
    loader_test = CIFAR10('./dataset', group='test', mode='reduced', batch=1024)
    record_train, record_test = utils.train(model, (loader_train, loader_test), total_epoch=100, learning_rate=1e-3, save_name='vgg_generalized', generalize=True)
    utils.plot(record_train, record_test, save_name='vgg_generalized')

    print('<experiment> generalized vgg16 on enhanced cifar10')
    model = VGGModel()
    loader_train = CIFAR10('./dataset', group='train', mode='enhanced', batch=1024)
    loader_test = CIFAR10('./dataset', group='test', mode='enhanced', batch=1024)
    record_train, record_test = utils.train(model, (loader_train, loader_test), total_epoch=100, learning_rate=1e-3, save_name='vgg_combined', generalize=True)
    utils.plot(record_train, record_test, save_name='vgg_combined')


if __name__ == '__main__':
    np.random.seed(0)
    jt.set_global_seed(0)
    jt.flags.use_cuda = 1
    execute()
