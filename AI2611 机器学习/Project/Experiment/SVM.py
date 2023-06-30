from tqdm import tqdm
import numpy as np
from kernel import CreateKernel


class SupportVectorMachine(object):
    def __init__(self, iteration=100, penalty=1.0, epsilon=1e-6, kernel=None):
        self.iteration = iteration
        self.penalty = penalty
        self.epsilon = epsilon
        if kernel is None:
            kernel = {'name': 'linear'}
        self.kernel = CreateKernel(kernel)
    
    def __compute_w(self):
        return (self.a * self.y) @ self.X

    def __compute_e(self, i):
        return (self.a * self.y) @ self.K[:, i] + self.b - self.y[i]
    
    def __select_j(self, i):
        j = np.random.randint(1, self.m)
        return j if j > i else j - 1
    
    def __step_forward(self, i):
        e_i = self.__compute_e(i)
        if ((self.a[i] > 0) and (e_i * self.y[i] > self.epsilon)) or ((self.a[i] < self.penalty) and (e_i * self.y[i] < -self.epsilon)):
            j = self.__select_j(i)
            e_j = self.__compute_e(j)
            a_i, a_j = np.copy(self.a[i]), np.copy(self.a[j])
            if self.y[i] == self.y[j]:
                L = max(0, a_i + a_j - self.penalty)
                H = min(self.penalty, a_i + a_j)
            else:
                L = max(0, a_j - a_i)
                H = min(self.penalty, self.penalty + a_j - a_i)
            if L == H:
                return False
            d = 2 * self.K[i, j] - self.K[i, i] - self.K[j, j]
            if d >= 0:
                return False
            self.a[j] = np.clip(a_j - self.y[j] * (e_i - e_j) / d, L, H)
            if np.abs(self.a[j] - a_j) < self.epsilon:
                return False
            self.a[i] = a_i + self.y[i] * self.y[j] * (a_j - self.a[j])
            b_i = self.b - e_i - self.y[i] * self.K[i, i] * (self.a[i] - a_i) - self.y[j] * self.K[j, i] * (self.a[j] - a_j)
            b_j = self.b - e_j - self.y[i] * self.K[i, j] * (self.a[i] - a_i) - self.y[j] * self.K[j, j] * (self.a[j] - a_j)
            if 0 < self.a[i] < self.penalty:
                self.b = b_i
            elif 0 < self.a[j] < self.penalty:
                self.b = b_j
            else:
                self.b = (b_i + b_j) / 2
            return True
        return False
    
    def setup(self, X, y):
        self.X, self.y = X, y
        self.m, self.n = X.shape
        self.b = 0.0
        self.a = np.zeros(self.m)
        self.K = np.zeros((self.m, self.m))
        for i in range(self.m):
            self.K[:, i] = self.kernel(X, X[i, :])
    
    def fit(self, X, y):
        self.setup(X, y)
        entire = True
        for _ in range(self.iteration):
            change = 0
            if entire:
                for i in range(self.m):
                    change += self.__step_forward(i)
            else:
                index = np.nonzero((0 < self.a) * (self.a < self.penalty))[0]
                for i in index:
                    change += self.__step_forward(i)
            if entire:
                entire = False
            elif change == 0:
                entire = True

    def predict(self, X):
        m = X.shape[0]
        y = np.zeros(m)
        for i in range(m):
            y[i] = np.sign((self.a * self.y) @ self.kernel(self.X, X[i, :]) + self.b)
        return y
    
    @property
    def weight(self):
        if self.kernel.name != 'linear':
            raise AttributeError('non-linear kernel')
        return self.__compute_w(), self.b


class SupportVectorClassifier(object):
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
                model = SupportVectorMachine(self.iteration, self.penalty, self.epsilon, self.kernel)
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
