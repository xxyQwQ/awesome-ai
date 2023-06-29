import numpy as np
import gd


m, n = 8, 6
X = np.array([
    [4, 1, 0, 4, 2, 0],
    [2, 4, 1, 1, 0, 2],
    [4, 4, 0, 4, 1, 4],
    [1, 0, 2, 3, 1, 2],
    [4, 4, 2, 2, 0, 1],
    [2, 2, 0, 1, 2, 4],
    [0, 1, 2, 1, 4, 2],
    [0, 0, 1, 0, 1, 3]
])
y = np.array([5, 0, 5, 0, 4, 2, 5, 3])


def f(w):
    return w.T @ X.T @ X @ w - 2 * y.T @ X @ w + y.T @ y


def fp(w):
    return 2 * (X.T @ X @ w - y.T @ X)


print('method: gradient descent')
w0 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
stepsize = 1.0 / 200.0
w_traces = gd.gd_const_ss(fp, w0, stepsize=stepsize, tol=1e-8)
w = w_traces[-1]
print('    iterations: {}'.format(len(w_traces) - 1))
if len(w_traces) - 1 == 100000:
    print('    state: diverge')
    w_traces = w_traces[0: 11]
else:
    print('    state: converge')
print('    solution: {}'.format(w))
print('    value: {:.6f}'.format(f(w)))


print('method: normal equation')
w = np.linalg.solve(X.T @ X, y.T @ X)
print('    solution: {}'.format(w))
print('    value: {:.6f}'.format(f(w)))
