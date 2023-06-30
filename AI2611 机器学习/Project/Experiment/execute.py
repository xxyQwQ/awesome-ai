import sys
import time
import numpy as np
from utils import Logger, ComputeAccuracy, BatchHOG
from dataset import MNIST, CIFAR10
from SVM import SupportVectorClassifier
from MKL import MultipleKernelClassifier


def convergence():
    sys.stdout = Logger('./output/convergence.txt')
    X_train, y_train = MNIST('./dataset/mnist_data/', group='train')
    X_test, y_test = MNIST('./dataset/mnist_data/', group='test')
    X_train, X_test = X_train.reshape(-1, 28*28), X_test.reshape(-1, 28*28)

    iteration_list = np.arange(5, 101, 5)
    for iteration in iteration_list:
        model = SupportVectorClassifier(iteration=iteration)
        model.fit(X_train, y_train)
        p_train, p_test = model.predict(X_train), model.predict(X_test)
        r_train, r_test = ComputeAccuracy(p_train, y_train), ComputeAccuracy(p_test, y_test)
        print('Iteration: {}, Train: {:.2%}, Test: {:.2%}'.format(iteration, r_train, r_test))


def ablation():
    sys.stdout = Logger('./output/ablation.txt')
    X_train, y_train = MNIST('./dataset/mnist_data/', group='train')
    X_test, y_test = MNIST('./dataset/mnist_data/', group='test')
    X_train, X_test = X_train.reshape(-1, 28*28), X_test.reshape(-1, 28*28)

    print('Kernel: Linear')
    penalty_list = [0.5 ** i for i in range(0, 8)]
    epsilon_list = [0.1 ** i for i in range(3, 7)]
    for penalty in penalty_list:
        for epsilon in epsilon_list:
            model = SupportVectorClassifier(iteration=100, penalty=penalty, epsilon=epsilon)
            model.fit(X_train, y_train)
            p_train, p_test = model.predict(X_train), model.predict(X_test)
            r_train, r_test = ComputeAccuracy(p_train, y_train), ComputeAccuracy(p_test, y_test)
            print('Penalty: {:.4f}, Epsilon: {:.1e}, Train: {:.2%}, Test: {:.2%}'.format(penalty, epsilon, r_train, r_test))
    
    print('Kernel: Polynomial')
    gamma_list = [0.5 ** i for i in range(0, 8)]
    degree_list = [i for i in range(1, 5)]
    for gamma in gamma_list:
        for degree in degree_list:
            kernel = {'name': 'polynomial', 'gamma': gamma, 'degree': degree}
            model = SupportVectorClassifier(iteration=100, kernel=kernel)
            model.fit(X_train, y_train)
            p_train, p_test = model.predict(X_train), model.predict(X_test)
            r_train, r_test = ComputeAccuracy(p_train, y_train), ComputeAccuracy(p_test, y_test)
            print('Gamma: {:.4f}, Degree: {}, Train: {:.2%}, Test: {:.2%}'.format(gamma, degree, r_train, r_test))
    
    print('Kernel: Gaussian')
    penalty_list = [0.5 ** i for i in range(0, 8)]
    gamma_list = [0.5 ** i for i in range(0, 4)]
    for penalty in penalty_list:
        for gamma in gamma_list:
            kernel = {'name': 'gaussian', 'gamma': gamma}
            model = SupportVectorClassifier(iteration=100, penalty=penalty, kernel=kernel)
            model.fit(X_train, y_train)
            p_train, p_test = model.predict(X_train), model.predict(X_test)
            r_train, r_test = ComputeAccuracy(p_train, y_train), ComputeAccuracy(p_test, y_test)
            print('Penalty: {:.4f}, Gamma: {:.4f}, Train: {:.2%}, Test: {:.2%}'.format(penalty, gamma, r_train, r_test))
    
    print('Kernel: Sigmoid')
    gamma_list = [0.5 ** i for i in range(0, 8)]
    bias_list = [-1.0, 0.0, 1.0]
    for gamma in gamma_list:
        for bias in bias_list:
            kernel = {'name': 'sigmoid', 'gamma': gamma, 'bias': bias}
            model = SupportVectorClassifier(iteration=100, kernel=kernel)
            model.fit(X_train, y_train)
            p_train, p_test = model.predict(X_train), model.predict(X_test)
            r_train, r_test = ComputeAccuracy(p_train, y_train), ComputeAccuracy(p_test, y_test)
            print('Gamma: {:.4f}, Bias: {:.2f}, Train: {:.2%}, Test: {:.2%}'.format(gamma, bias, r_train, r_test))


def precision():
    sys.stdout = Logger('./output/precision.txt')

    print('Dataset: MNIST')
    X_train, y_train = MNIST('./dataset/mnist_data/', group='train')
    X_test, y_test = MNIST('./dataset/mnist_data/', group='test')
    X_train, X_test = X_train.reshape(-1, 28*28), X_test.reshape(-1, 28*28)

    model = SupportVectorClassifier(iteration=100, penalty=0.05)
    model.fit(X_train, y_train)
    p_train, p_test = model.predict(X_train), model.predict(X_test)
    r_train, r_test = ComputeAccuracy(p_train, y_train), ComputeAccuracy(p_test, y_test)
    print('Kernel: Linear, Train: {:.2%}, Test: {:.2%}'.format(r_train, r_test))

    kernel = {'name': 'polynomial', 'gamma': 1.0, 'degree': 2}
    model = SupportVectorClassifier(iteration=100, kernel=kernel)
    model.fit(X_train, y_train)
    p_train, p_test = model.predict(X_train), model.predict(X_test)
    r_train, r_test = ComputeAccuracy(p_train, y_train), ComputeAccuracy(p_test, y_test)
    print('Kernel: Polynomial, Train: {:.2%}, Test: {:.2%}'.format(r_train, r_test))

    kernel = {'name': 'gaussian', 'gamma': 0.03}
    model = SupportVectorClassifier(iteration=100, kernel=kernel)
    model.fit(X_train, y_train)
    p_train, p_test = model.predict(X_train), model.predict(X_test)
    r_train, r_test = ComputeAccuracy(p_train, y_train), ComputeAccuracy(p_test, y_test)
    print('Kernel: Gaussian, Train: {:.2%}, Test: {:.2%}'.format(r_train, r_test))

    kernel = {'name': 'sigmoid', 'gamma': 0.01, 'bias': -1.0}
    model = SupportVectorClassifier(iteration=100, kernel=kernel)
    model.fit(X_train, y_train)
    p_train, p_test = model.predict(X_train), model.predict(X_test)
    r_train, r_test = ComputeAccuracy(p_train, y_train), ComputeAccuracy(p_test, y_test)
    print('Kernel: Sigmoid, Train: {:.2%}, Test: {:.2%}'.format(r_train, r_test))

    kernel = [
        {'name': 'linear'},
        {'name': 'polynomial', 'gamma': 1.0, 'degree': 2},
        {'name': 'gaussian', 'gamma': 0.03},
        {'name': 'sigmoid', 'gamma': 0.01, 'bias': -1.0},
    ]
    model = MultipleKernelClassifier(iteration=100, kernel=kernel)
    model.fit(X_train, y_train)
    p_train, p_test = model.predict(X_train), model.predict(X_test)
    r_train, r_test = ComputeAccuracy(p_train, y_train), ComputeAccuracy(p_test, y_test)
    print('Kernel: Conbined, Train: {:.2%}, Test: {:.2%}'.format(r_train, r_test))
    
    print('Dataset: CIFAR10')
    X_train, y_train = CIFAR10('./dataset/cifar-10-batches-py/', group='train')
    X_test, y_test = CIFAR10('./dataset/cifar-10-batches-py/', group='test')
    X_train, X_test = X_train.reshape(-1, 3*32*32), X_test.reshape(-1, 3*32*32)

    model = SupportVectorClassifier(iteration=100, penalty=0.1)
    model.fit(X_train, y_train)
    p_train, p_test = model.predict(X_train), model.predict(X_test)
    r_train, r_test = ComputeAccuracy(p_train, y_train), ComputeAccuracy(p_test, y_test)
    print('Kernel: Linear, Train: {:.2%}, Test: {:.2%}'.format(r_train, r_test))

    kernel = {'name': 'polynomial', 'gamma': 0.05, 'degree': 3}
    model = SupportVectorClassifier(iteration=100, kernel=kernel)
    model.fit(X_train, y_train)
    p_train, p_test = model.predict(X_train), model.predict(X_test)
    r_train, r_test = ComputeAccuracy(p_train, y_train), ComputeAccuracy(p_test, y_test)
    print('Kernel: Polynomial, Train: {:.2%}, Test: {:.2%}'.format(r_train, r_test))

    kernel = {'name': 'gaussian', 'gamma': 0.03}
    model = SupportVectorClassifier(iteration=100, kernel=kernel)
    model.fit(X_train, y_train)
    p_train, p_test = model.predict(X_train), model.predict(X_test)
    r_train, r_test = ComputeAccuracy(p_train, y_train), ComputeAccuracy(p_test, y_test)
    print('Kernel: Gaussian, Train: {:.2%}, Test: {:.2%}'.format(r_train, r_test))

    kernel = {'name': 'sigmoid', 'gamma': 0.01, 'bias': -1.0}
    model = SupportVectorClassifier(iteration=100, kernel=kernel)
    model.fit(X_train, y_train)
    p_train, p_test = model.predict(X_train), model.predict(X_test)
    r_train, r_test = ComputeAccuracy(p_train, y_train), ComputeAccuracy(p_test, y_test)
    print('Kernel: Sigmoid, Train: {:.2%}, Test: {:.2%}'.format(r_train, r_test))

    kernel = [
        {'name': 'linear'},
        {'name': 'polynomial', 'gamma': 0.05, 'degree': 3},
        {'name': 'gaussian', 'gamma': 0.03},
        {'name': 'sigmoid', 'gamma': 0.01, 'bias': -1.0},
    ]
    model = MultipleKernelClassifier(iteration=100, kernel=kernel)
    model.fit(X_train, y_train)
    p_train, p_test = model.predict(X_train), model.predict(X_test)
    r_train, r_test = ComputeAccuracy(p_train, y_train), ComputeAccuracy(p_test, y_test)
    print('Kernel: Conbined, Train: {:.2%}, Test: {:.2%}'.format(r_train, r_test))

    print('Dataset: CIFAR10-HOG')
    X_train, y_train = CIFAR10('./dataset/cifar-10-batches-py/', group='train')
    X_test, y_test = CIFAR10('./dataset/cifar-10-batches-py/', group='test')
    X_train, X_test = BatchHOG(X_train, partition=16), BatchHOG(X_test, partition=16)

    model = SupportVectorClassifier(iteration=100, penalty=0.1)
    model.fit(X_train, y_train)
    p_train, p_test = model.predict(X_train), model.predict(X_test)
    r_train, r_test = ComputeAccuracy(p_train, y_train), ComputeAccuracy(p_test, y_test)
    print('Kernel: Linear, Train: {:.2%}, Test: {:.2%}'.format(r_train, r_test))

    kernel = {'name': 'polynomial', 'gamma': 0.05, 'degree': 3}
    model = SupportVectorClassifier(iteration=100, kernel=kernel)
    model.fit(X_train, y_train)
    p_train, p_test = model.predict(X_train), model.predict(X_test)
    r_train, r_test = ComputeAccuracy(p_train, y_train), ComputeAccuracy(p_test, y_test)
    print('Kernel: Polynomial, Train: {:.2%}, Test: {:.2%}'.format(r_train, r_test))

    kernel = {'name': 'gaussian', 'gamma': 0.03}
    model = SupportVectorClassifier(iteration=100, kernel=kernel)
    model.fit(X_train, y_train)
    p_train, p_test = model.predict(X_train), model.predict(X_test)
    r_train, r_test = ComputeAccuracy(p_train, y_train), ComputeAccuracy(p_test, y_test)
    print('Kernel: Gaussian, Train: {:.2%}, Test: {:.2%}'.format(r_train, r_test))

    kernel = {'name': 'sigmoid', 'gamma': 0.01, 'bias': -1.0}
    model = SupportVectorClassifier(iteration=100, kernel=kernel)
    model.fit(X_train, y_train)
    p_train, p_test = model.predict(X_train), model.predict(X_test)
    r_train, r_test = ComputeAccuracy(p_train, y_train), ComputeAccuracy(p_test, y_test)
    print('Kernel: Sigmoid, Train: {:.2%}, Test: {:.2%}'.format(r_train, r_test))

    kernel = [
        {'name': 'linear'},
        {'name': 'polynomial', 'gamma': 0.05, 'degree': 3},
        {'name': 'gaussian', 'gamma': 0.03},
        {'name': 'sigmoid', 'gamma': 0.01, 'bias': -1.0},
    ]
    model = MultipleKernelClassifier(iteration=100, kernel=kernel)
    model.fit(X_train, y_train)
    p_train, p_test = model.predict(X_train), model.predict(X_test)
    r_train, r_test = ComputeAccuracy(p_train, y_train), ComputeAccuracy(p_test, y_test)
    print('Kernel: Conbined, Train: {:.2%}, Test: {:.2%}'.format(r_train, r_test))


def efficiency():
    sys.stdout = Logger('./output/efficiency.txt')
    X_train, y_train = MNIST('./dataset/mnist_data/', group='train')
    X_test, _ = MNIST('./dataset/mnist_data/', group='test')
    X_train, X_test = X_train.reshape(-1, 28*28), X_test.reshape(-1, 28*28)

    model = SupportVectorClassifier(iteration=100, penalty=0.05)
    start = time.time()
    model.fit(X_train, y_train)
    finish = time.time()
    t_train = finish - start
    start = time.time()
    model.predict(X_test)
    finish = time.time()
    t_test = finish - start
    print('Kernel: Linear, Train: {:.2f}, Test: {:.2f}'.format(t_train, t_test))

    kernel = {'name': 'polynomial', 'gamma': 1.0, 'degree': 2}
    model = SupportVectorClassifier(iteration=100, kernel=kernel)
    start = time.time()
    model.fit(X_train, y_train)
    finish = time.time()
    t_train = finish - start
    start = time.time()
    model.predict(X_test)
    finish = time.time()
    t_test = finish - start
    print('Kernel: Polynomial, Train: {:.2f}, Test: {:.2f}'.format(t_train, t_test))

    kernel = {'name': 'gaussian', 'gamma': 0.03}
    model = SupportVectorClassifier(iteration=100, kernel=kernel)
    start = time.time()
    model.fit(X_train, y_train)
    finish = time.time()
    t_train = finish - start
    start = time.time()
    model.predict(X_test)
    finish = time.time()
    t_test = finish - start
    print('Kernel: Gaussian, Train: {:.2f}, Test: {:.2f}'.format(t_train, t_test))

    kernel = {'name': 'sigmoid', 'gamma': 0.01, 'bias': -1.0}
    model = SupportVectorClassifier(iteration=100, kernel=kernel)
    start = time.time()
    model.fit(X_train, y_train)
    finish = time.time()
    t_train = finish - start
    start = time.time()
    model.predict(X_test)
    finish = time.time()
    t_test = finish - start
    print('Kernel: Sigmoid, Train: {:.2f}, Test: {:.2f}'.format(t_train, t_test))

    kernel = [
        {'name': 'linear'},
        {'name': 'polynomial', 'gamma': 1.0, 'degree': 2},
        {'name': 'gaussian', 'gamma': 0.03},
        {'name': 'sigmoid', 'gamma': 0.01, 'bias': -1.0},
    ]
    model = MultipleKernelClassifier(iteration=100, kernel=kernel)
    start = time.time()
    model.fit(X_train, y_train)
    finish = time.time()
    t_train = finish - start
    start = time.time()
    model.predict(X_test)
    finish = time.time()
    t_test = finish - start
    print('Kernel: Conbined, Train: {:.2f}, Test: {:.2f}'.format(t_train, t_test))


def combination():
    sys.stdout = Logger('./output/combination.txt')
    X_train, y_train = MNIST('./dataset/mnist_data/', group='train')
    X_test, y_test = MNIST('./dataset/mnist_data/', group='test')
    X_train, X_test = X_train.reshape(-1, 28*28), X_test.reshape(-1, 28*28)

    kernel = [
        {'name': 'linear'},
        {'name': 'polynomial', 'gamma': 1.0, 'degree': 2},
    ]
    model = MultipleKernelClassifier(iteration=100, kernel=kernel)
    model.fit(X_train, y_train)
    p_train, p_test = model.predict(X_train), model.predict(X_test)
    r_train, r_test = ComputeAccuracy(p_train, y_train), ComputeAccuracy(p_test, y_test)
    print('kernel: Linear+Polynomial, Train: {:.2%}, Test: {:.2%}'.format(r_train, r_test))

    kernel = [
        {'name': 'linear'},
        {'name': 'gaussian', 'gamma': 0.03},
    ]
    model = MultipleKernelClassifier(iteration=100, kernel=kernel)
    model.fit(X_train, y_train)
    p_train, p_test = model.predict(X_train), model.predict(X_test)
    r_train, r_test = ComputeAccuracy(p_train, y_train), ComputeAccuracy(p_test, y_test)
    print('kernel: Linear+Gaussian, Train: {:.2%}, Test: {:.2%}'.format(r_train, r_test))

    kernel = [
        {'name': 'linear'},
        {'name': 'sigmoid', 'gamma': 0.01, 'bias': -1.0},
    ]
    model = MultipleKernelClassifier(iteration=100, kernel=kernel)
    model.fit(X_train, y_train)
    p_train, p_test = model.predict(X_train), model.predict(X_test)
    r_train, r_test = ComputeAccuracy(p_train, y_train), ComputeAccuracy(p_test, y_test)
    print('kernel: Linear+Sigmoid, Train: {:.2%}, Test: {:.2%}'.format(r_train, r_test))

    kernel = [
        {'name': 'polynomial', 'gamma': 1.0, 'degree': 2},
        {'name': 'gaussian', 'gamma': 0.03},
    ]
    model = MultipleKernelClassifier(iteration=100, kernel=kernel)
    model.fit(X_train, y_train)
    p_train, p_test = model.predict(X_train), model.predict(X_test)
    r_train, r_test = ComputeAccuracy(p_train, y_train), ComputeAccuracy(p_test, y_test)
    print('kernel: Polynomial+Gaussian, Train: {:.2%}, Test: {:.2%}'.format(r_train, r_test))

    kernel = [
        {'name': 'polynomial', 'gamma': 1.0, 'degree': 2},
        {'name': 'sigmoid', 'gamma': 0.01, 'bias': -1.0},
    ]
    model = MultipleKernelClassifier(iteration=100, kernel=kernel)
    model.fit(X_train, y_train)
    p_train, p_test = model.predict(X_train), model.predict(X_test)
    r_train, r_test = ComputeAccuracy(p_train, y_train), ComputeAccuracy(p_test, y_test)
    print('kernel: Polynomial+Sigmoid, Train: {:.2%}, Test: {:.2%}'.format(r_train, r_test))

    kernel = [
        {'name': 'gaussian', 'gamma': 0.03},
        {'name': 'sigmoid', 'gamma': 0.01, 'bias': -1.0},
    ]
    model = MultipleKernelClassifier(iteration=100, kernel=kernel)
    model.fit(X_train, y_train)
    p_train, p_test = model.predict(X_train), model.predict(X_test)
    r_train, r_test = ComputeAccuracy(p_train, y_train), ComputeAccuracy(p_test, y_test)
    print('kernel: Gaussian+Sigmoid, Train: {:.2%}, Test: {:.2%}'.format(r_train, r_test))

    kernel = [
        {'name': 'linear'},
        {'name': 'polynomial', 'gamma': 1.0, 'degree': 2},
        {'name': 'gaussian', 'gamma': 0.03},
    ]
    model = MultipleKernelClassifier(iteration=100, kernel=kernel)
    model.fit(X_train, y_train)
    p_train, p_test = model.predict(X_train), model.predict(X_test)
    r_train, r_test = ComputeAccuracy(p_train, y_train), ComputeAccuracy(p_test, y_test)
    print('kernel: Linear+Polynomial+Gaussian, Train: {:.2%}, Test: {:.2%}'.format(r_train, r_test))

    kernel = [
        {'name': 'linear'},
        {'name': 'polynomial', 'gamma': 1.0, 'degree': 2},
        {'name': 'sigmoid', 'gamma': 0.01, 'bias': -1.0},
    ]
    model = MultipleKernelClassifier(iteration=100, kernel=kernel)
    model.fit(X_train, y_train)
    p_train, p_test = model.predict(X_train), model.predict(X_test)
    r_train, r_test = ComputeAccuracy(p_train, y_train), ComputeAccuracy(p_test, y_test)
    print('kernel: Linear+Polynomial+Sigmoid, Train: {:.2%}, Test: {:.2%}'.format(r_train, r_test))

    kernel = [
        {'name': 'linear'},
        {'name': 'gaussian', 'gamma': 0.03},
        {'name': 'sigmoid', 'gamma': 0.01, 'bias': -1.0},
    ]
    model = MultipleKernelClassifier(iteration=100, kernel=kernel)
    model.fit(X_train, y_train)
    p_train, p_test = model.predict(X_train), model.predict(X_test)
    r_train, r_test = ComputeAccuracy(p_train, y_train), ComputeAccuracy(p_test, y_test)
    print('kernel: Linear+Gaussian+Sigmoid, Train: {:.2%}, Test: {:.2%}'.format(r_train, r_test))

    kernel = [
        {'name': 'polynomial', 'gamma': 1.0, 'degree': 2},
        {'name': 'gaussian', 'gamma': 0.03},
        {'name': 'sigmoid', 'gamma': 0.01, 'bias': -1.0},
    ]
    model = MultipleKernelClassifier(iteration=100, kernel=kernel)
    model.fit(X_train, y_train)
    p_train, p_test = model.predict(X_train), model.predict(X_test)
    r_train, r_test = ComputeAccuracy(p_train, y_train), ComputeAccuracy(p_test, y_test)
    print('kernel: Polynomial+Gaussian+Sigmoid, Train: {:.2%}, Test: {:.2%}'.format(r_train, r_test))

    kernel = [
        {'name': 'linear'},
        {'name': 'polynomial', 'gamma': 1.0, 'degree': 2},
        {'name': 'gaussian', 'gamma': 0.03},
        {'name': 'sigmoid', 'gamma': 0.01, 'bias': -1.0},
    ]
    model = MultipleKernelClassifier(iteration=100, kernel=kernel)
    model.fit(X_train, y_train)
    p_train, p_test = model.predict(X_train), model.predict(X_test)
    r_train, r_test = ComputeAccuracy(p_train, y_train), ComputeAccuracy(p_test, y_test)
    print('kernel: Linear+Polynomial+Gaussian+Sigmoid, Train: {:.2%}, Test: {:.2%}'.format(r_train, r_test))


def main():
    convergence()
    ablation()
    precision()
    efficiency()
    combination()


if __name__ == '__main__':
    main()
