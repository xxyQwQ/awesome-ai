import numpy as np
import newton
import utils
import matplotlib.pyplot as plt


X = np.array([[1, 1.5],
              [1.2, 2.5],
              [1, 3.5],
              [2, 2.25],
              [1.8, 3],
              [2.5, 4],
              [3, 1.9],
              [1.5, .5],
              [2.5, .8],
              [2.8, .3],
              [3.2, .3],
              [3, .8],
              [3.8, 1],
              [4, 2],
              [1.8, 1.8]])
y = np.append(np.ones((7,)), -np.ones((8,)))
Xy = X * y.reshape((-1, 1))


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def sigmoid_p(z):
    s = sigmoid(z)
    return s * (1 - s)


def f(w):
    return -np.sum(np.log(sigmoid(Xy @ w)))


def fp(w):
    return (-1 * (Xy.T @ (1 - sigmoid(Xy @ w)).reshape((-1, 1)))).reshape(-1)


def fpp(w):
    P = sigmoid_p(Xy @ w).reshape((-1, 1)) * X
    return P.T @ X


def f_2d(w1, w2):
    res = 0
    for i in range(Xy.shape[0]):
        res -= np.log(sigmoid(Xy[i, 0]*w1+Xy[i, 1]*w2))
    return res


def gap(w):
    return f(w) - fs


w0s = [np.array([-1.5, 1.0]), np.array([1.0, 1.0])]
path = './figures/p1/'


# Pure
w0 = w0s[0]
print("pure Newton's method with initial point", w0)
w_traces = newton.newton(fp, fpp, w0, maxiter=100)
ws = w_traces[-1]
fs = f(ws)
if len(w_traces) - 1 == 100:
    print('\tstate: diverge')
else:
    print('\tstate: converge')
print('\titerations:', len(w_traces) - 1)
print('\tsolution:', ws)
print('\tvalue:', fs)
utils.plot_traces_2d(f_2d, w_traces, path + 'nt_traces.png')
utils.plot(gap, w_traces, path + 'nt_gap.png')


# Damped
for id in range(len(w0s)):
    w0 = w0s[id]
    print("damped Newton's method with initial point", w0)
    w_traces, stepsize_traces, num_iter_inner = newton.damped_newton(
        f, fp, fpp, w0, alpha=0.1, beta=0.7, maxiter=100)
    ws = w_traces[-1]
    fs = f(ws)
    if len(w_traces) - 1 == 100:
        print('\tstate: diverge')
    else:
        print('\tstate: converge')
    print('\titerations (outer loop):', len(w_traces) - 1)
    print('\titerations (inner loop):', num_iter_inner)
    print('\tsolution:', ws)
    print('\tvalue:', fs)
    utils.plot_traces_2d(f_2d, w_traces, path + 'dnt_traces_{}.png'.format(id))
    utils.plot(gap, w_traces, path + 'dnt_gap_{}.png'.format(id))
    fig = plt.figure(figsize=(3.5, 2.5))
    plt.plot(stepsize_traces, '-o', color='blue')
    plt.xlabel('iteration (k)')
    plt.ylabel('stepsize')
    plt.tight_layout(pad=0.1)
    fig.savefig(path+'dnt_ss_{}.png'.format(id))
