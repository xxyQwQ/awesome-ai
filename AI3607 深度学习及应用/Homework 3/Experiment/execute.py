import sys
import numpy as np
import jittor as jt
from dataset import MNIST
from model import RNNModel, LSTMModel
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

    print('<experiment> rnn on mnist')
    model = RNNModel()
    loader_train = MNIST('./dataset', group='train', mode='origin', batch=1024)
    loader_test = MNIST('./dataset', group='test', mode='origin', batch=1024)
    record_train, record_test = utils.train(model, (loader_train, loader_test), total_epoch=100, learning_rate=1e-3, save_name='rnn_origin')
    utils.plot(record_train, record_test, save_name='rnn_origin')

    print('<experiment> lstm on mnist')
    model = LSTMModel()
    loader_train = MNIST('./dataset', group='train', mode='origin', batch=1024)
    loader_test = MNIST('./dataset', group='test', mode='origin', batch=1024)
    record_train, record_test = utils.train(model, (loader_train, loader_test), total_epoch=100, learning_rate=1e-3, save_name='lstm_origin')
    utils.plot(record_train, record_test, save_name='lstm_origin')

    print('<experiment> lstm on reduced mnist')
    model = LSTMModel()
    loader_train = MNIST('./dataset', group='train', mode='reduced', batch=1024)
    loader_test = MNIST('./dataset', group='test', mode='reduced', batch=1024)
    record_train, record_test = utils.train(model, (loader_train, loader_test), total_epoch=100, learning_rate=1e-3, save_name='lstm_reduced')
    utils.plot(record_train, record_test, save_name='lstm_reduced')

    print('<experiment> lstm on enhanced mnist')
    model = LSTMModel()
    loader_train = MNIST('./dataset', group='train', mode='enhanced', batch=1024)
    loader_test = MNIST('./dataset', group='test', mode='enhanced', batch=1024)
    record_train, record_test = utils.train(model, (loader_train, loader_test), total_epoch=100, learning_rate=1e-3, save_name='lstm_enhanced')
    utils.plot(record_train, record_test, save_name='lstm_enhanced')

    print('<experiment> generalized lstm on reduced mnist')
    model = LSTMModel()
    loader_train = MNIST('./dataset', group='train', mode='reduced', batch=1024)
    loader_test = MNIST('./dataset', group='test', mode='reduced', batch=1024)
    record_train, record_test = utils.train(model, (loader_train, loader_test), total_epoch=100, learning_rate=1e-3, save_name='lstm_generalized', generalize=True)
    utils.plot(record_train, record_test, save_name='lstm_generalized')

    print('<experiment> generalized lstm on enhanced mnist')
    model = LSTMModel()
    loader_train = MNIST('./dataset', group='train', mode='enhanced', batch=1024)
    loader_test = MNIST('./dataset', group='test', mode='enhanced', batch=1024)
    record_train, record_test = utils.train(model, (loader_train, loader_test), total_epoch=100, learning_rate=1e-3, save_name='lstm_combined', generalize=True)
    utils.plot(record_train, record_test, save_name='lstm_combined')


if __name__ == '__main__':
    np.random.seed(0)
    jt.set_global_seed(0)
    jt.flags.use_cuda = 1
    execute()
