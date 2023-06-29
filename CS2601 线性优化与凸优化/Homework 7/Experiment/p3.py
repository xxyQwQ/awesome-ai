import numpy as np
import ista
import utils


X = np.array([[1, 2, 0], [0, 1, 0]]).T
y = np.array([2, 3, 2])
lambds = [2.0, 0.1, 8.0]
stepsize = 0.1
path = './figures/p3/'


def f(w):
    return 0.5 * np.sum((X @ w - y) ** 2) + lambd * np.linalg.norm(w, ord=1)


def f_2d(w1, w2):
    return 0.5 * Q[0, 0] * w1**2 + 0.5 * Q[1, 1] * w2**2 + Q[0, 1] * w1 * w2 \
        - b[0] * w1 - b[1] * w2 - 0.5 * c + lambd * (np.abs(w1) + np.abs(w2))


def gap(w):
    return f(w) - f_value


for id in range(len(lambds)):
    lambd = lambds[id]
    print("ISTA algorithm with lambda =", lambd)
    w0 = 0.5 * np.ones(2)
    w_traces = ista.ista(X, y, lambd, w0, stepsize=stepsize,
                         tol=1e-9, maxiter=10000)
    f_value = f(w_traces[-1])
    if len(w_traces) - 1 == 10000:
        print('\tstate: diverge')
    else:
        print('\tstate: converge')
    print('\titerations:', len(w_traces) - 1)
    print('\tsolution:', w_traces[-1])
    print('\tvalue:', f_value)
    Q = X.T @ X
    b = X.T @ y
    c = y @ y
    utils.plot_traces_2d(f_2d, w_traces, path +
                         f'ista_traces_lambda{lambd}.png')
    utils.plot(gap, w_traces, path + f'ista_gap_lambda{lambd}.png')
