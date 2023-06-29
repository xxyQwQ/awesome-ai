import numpy as np
import proj_gd as gd
import utils
import matplotlib.pyplot as plt
import matplotlib.patches as mp


X = np.array([[2, 0, 1], [0, 2, 1]]).T
y = np.array([3, 1, 2])
t = 1


def f(w):
    return 0.5 * np.sum((X @ w - y)**2)


def fp(w):
    return X.T @ (X @ w - y)


def proj(x, t=1):
    if np.linalg.norm(x, ord=1) <= t:
        return x
    y = np.sort(np.abs(x))[::-1]
    mu = (np.cumsum(y) - t) / np.arange(1, len(y) + 1)
    x_base = np.maximum(np.abs(x) - mu[np.where(mu <= y)[0][-1]], 0)
    return np.sign(x) * x_base


w0 = np.array([-1, 0.5])
stepsize = 0.1
w_traces, y_traces = gd.proj_gd(fp, proj, w0, stepsize=stepsize, tol=1e-8)
f_value = f(w_traces[-1])
print('iterations: {}'.format(len(w_traces) - 1))
if len(w_traces) - 1 == 100000:
    print('state: diverge')
    w_traces = w_traces[0: 11]
else:
    print('state: converge')
print('solution: {}'.format(w_traces[-1]))
print('value: {:.6f}'.format(f_value))


Q = X.T @ X
b = X.T @ y
c = y @ y


def f_2d(w1, w2):
    return 0.5 * Q[0, 0] * w1 ** 2 + 0.5 * Q[1, 1] * w2 ** 2 + Q[0, 1] * w1 * w2 \
        - b[0] * w1 - b[1] * w2 - 0.5 * c


def gap(w):
    return f(w) - f_value


path = './figures/'
feasible_set = mp.Polygon(
    [(-t, 0), (0, t), (t, 0), (0, -t)], alpha=0.5, color='y')
utils.plot_traces_2d(f_2d, w_traces, y_traces, feasible_set,
                     path+f'lasso_traces_t{t}.png')
utils.plot(gap, w_traces, path+f'lasso_gap_t{t}.png')
