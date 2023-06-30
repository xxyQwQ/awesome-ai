from tqdm import tqdm
import numpy as np
from utils import MatrixSimilarity
from kernel import CreateKernel
from SVM import SupportVectorMachine


class MultipleKernelLearning(SupportVectorMachine):
    def __init__(self, iteration=100, penalty=1.0, epsilon=1e-6, kernel=None):
        self.iteration = iteration
        self.penalty = penalty
        self.epsilon = epsilon
        if kernel is None:
            kernel = [{'name': 'linear'}]
        self.component = []
        for entry in kernel:
            self.component.append(CreateKernel(entry))
    
    def setup(self, X, y):
        self.X, self.y = X, y
        self.m, self.n = X.shape
        self.b = 0.0
        self.a = np.zeros(self.m)
        self.c = len(self.component)
        Y = np.outer(y, y)
        A = np.zeros(self.c)
        K = np.zeros((self.c, self.m, self.m))
        for i in range(self.c):
            for j in range(self.m):
                K[i, :, j] = self.component[i](X, X[j, :])
            A[i] = MatrixSimilarity(K[i], Y)
        self.w = A / np.sum(A)
        self.K = np.sum([self.w[i] * K[i] for i in range(self.c)], axis=0)
    
    def kernel(self, X, y):
        return np.sum([self.w[i] * self.component[i](X, y) for i in range(self.c)], axis=0)

    def fit(self, X, y):
        super().fit(X, y)

    def predict(self, X):
        return super().predict(X)


class MultipleKernelClassifier(object):
    def __init__(self, iteration=100, penalty=1.0, epsilon=1e-6, kernel=None):
        self.iteration = iteration
        self.penalty = penalty
        self.epsilon = epsilon
        self.kernel = kernel
        self.classifier = []

    def __build_model(self, y):
        self.label = np.unique(y)
        for i in range(len(self.label)):
            for j in range(i+1, len(self.label)):
                model = MultipleKernelLearning(self.iteration, self.penalty, self.epsilon, self.kernel)
                self.classifier.append((i, j, model))

    def fit(self, X, y):
        self.__build_model(y)
        for i, j, model in tqdm(self.classifier):
            index = np.where((y == self.label[i]) | (y == self.label[j]))[0]
            X_ij, y_ij = X[index], np.where(y[index] == self.label[i], -1, 1)
            model.fit(X_ij, y_ij)
    
    def predict(self, X):
        vote = np.zeros((X.shape[0], len(self.label)))
        for i, j, model in tqdm(self.classifier):
            y = model.predict(X)
            vote[np.where(y == -1)[0], i] += 1
            vote[np.where(y == 1)[0], j] += 1
        return self.label[np.argmax(vote, axis=1)]
