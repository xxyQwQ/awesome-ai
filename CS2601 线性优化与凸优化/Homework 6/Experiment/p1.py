import numpy as np
import gd
import utils


gamma = 10
Q = np.diag([1, gamma])


def f(x):
    return 0.5 * x.T @ Q @ x


def fp(x):
    return Q @ x


def f_2d(x1, x2):
    return 0.5 * 1 * x1**2 + 0.5 * gamma * x2**2


x0 = np.array([1.0, 1.0])
stepsizes = [0.22, 0.1, 0.01, 0.001]
print('gamma = {:.3f}'.format(gamma))
for stepsize in stepsizes:
    x_traces = gd.gd_const_ss(fp, x0, stepsize=stepsize)
    print(
        '    stepsize = {:.3f}'.format(stepsize))
    print('        iterations: {}'.format(len(x_traces) - 1))
    if len(x_traces) - 1 == 100000:
        print('        state: diverge')
        x_traces = x_traces[0: 11]
    else:
        print('        state: converge')
    utils.plot_traces_2d(
        f_2d, x_traces, './figures/gd_traces_gamma{:.3f}_ss{:.3f}.jpg'.format(gamma, stepsize))
    utils.plot(
        f, x_traces, './figures/gd_f_gamma{:.3f}_ss{:.3f}.jpg'.format(gamma, stepsize))


stepsize = 1
gammas = [1, 0.1, 0.01, 0.001]
print('stepsize = {:.3f}'.format(stepsize))
for gamma in gammas:
    Q = np.diag([1, gamma])
    x_traces = gd.gd_const_ss(fp, x0, stepsize=stepsize)
    print(
        '    gamma = {:.3f}'.format(gamma))
    print('        iterations: {}'.format(len(x_traces) - 1))
    if len(x_traces) - 1 == 100000:
        print('        state: diverge')
        x_traces = x_traces[0: 11]
    else:
        print('        state: converge')
