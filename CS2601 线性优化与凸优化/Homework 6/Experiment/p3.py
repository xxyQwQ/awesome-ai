import numpy as np
import matplotlib.pyplot as plt
import gd
import utils


X = np.array([
    [3, 1.5],
    [3.2, 2.5],
    [3, 3.5],
    [2, 2.25],
    [3.8, 3],
    [1.5, 4],
    [1, 1.9],
    [4.5, .5],
    [3.5, .8],
    [3.8, .3],
    [4.2, .3],
    [1, .8],
    [3.8, 1],
    [4, 2],
    [5.8, 1.8]
])
y = np.append(np.ones((7,)), -np.ones((8,)))
X = np.append(X, np.ones((15, 1)), axis=1)
Xy = X * y.reshape((-1, 1))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fp(w):
    return (-1 * (Xy.T @ (1 - sigmoid(Xy @ w)).reshape((-1, 1)))).reshape(-1)


def err(w):
    return np.linalg.norm(fp(w))


# minimize f by gradient descent
w0 = np.array([0, 0, 0])
stepsize = 1e-2
w_traces = gd.gd_const_ss(fp, w0, stepsize=stepsize)


# compute the accuracy on the training set
w = w_traces[-1]
print('iterations: {}'.format(len(w_traces) - 1))
if len(w_traces) - 1 == 100000:
    print('state: diverge')
    w_traces = w_traces[0: 11]
else:
    print('state: converge')
print('solution: {}'.format(w))
y_hat = 2 * (X @ w > 0) - 1
accuracy = sum(y_hat == y) / float(len(y))
print('accuracy: {:.2f}%'.format(accuracy * 100))


# visualization
plt.figure(figsize=(4, 4))
plt.scatter(*zip(*X[y > 0, 0:2]), c='r', marker='x')
plt.scatter(*zip(*X[y < 0, 0:2]), c='g', marker='o')
x1 = np.array([min(X[:, 0]), max(X[:, 0])])
x2 = -(w[0] * x1 + w[2]) / w[1]
plt.plot(x1, x2, 'b-')
plt.xlabel('x1')
plt.ylabel('x2')
plt.tight_layout()
plt.savefig('./figures/p3_visualization.jpg')
plt.figure(figsize=(4, 4))
utils.plot(err, w_traces, './figures/p3_gradient_descent.jpg')
