import numpy as np


class LinearKernel(object):
    def __init__(self):
        self.name = 'linear'
    
    def __call__(self, X, y):
        return X @ y.T


class PolynomialKernel(object):
    def __init__(self, gamma=1.0, degree=3):
        self.name = 'polynomial'
        self.gamma = gamma
        self.degree = degree
    
    def __call__(self, X, y):
        return np.power(self.gamma * (X @ y.T) + 1, self.degree)


class GaussianKernel(object):
    def __init__(self, gamma=1.0):
        self.name = 'gaussian'
        self.gamma = gamma
    
    def __call__(self, X, y):
        return np.exp(-self.gamma * np.sum(np.square(X - y), axis=1))


class SigmoidKernel(object):
    def __init__(self, gamma=1.0, bias=0.0):
        self.name = 'sigmoid'
        self.gamma = gamma
        self.bias = bias
    
    def __call__(self, X, y):
        return np.tanh(self.gamma * (X @ y.T) + self.bias)


def CreateKernel(entry):
    if entry['name'] == 'linear':
        return LinearKernel()
    elif entry['name'] == 'polynomial':
        return PolynomialKernel(entry['gamma'], entry['degree'])
    elif entry['name'] == 'gaussian':
        return GaussianKernel(entry['gamma'])
    elif entry['name'] == 'sigmoid':
        return SigmoidKernel(entry['gamma'], entry['bias'])
    raise AttributeError('invalid kernel')
