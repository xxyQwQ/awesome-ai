import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp


def plot_traces_2d(f_2d, x_traces, filename):
    fig = plt.figure(figsize=(5, 4))

    plt.plot(*zip(*x_traces), '-o', color='red')

    x1, x2 = zip(*x_traces)
    x1 = np.arange(min(x1)-.2, max(x1)+.2, 0.01)
    x2 = np.arange(min(x2)-.2, max(x2)+.2, 0.01)
    x1, x2 = np.meshgrid(x1, x2)
    plt.contour(x1, x2, f_2d(x1, x2), 20, colors='blue')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.tight_layout()

    fig.savefig(filename)


def plot(f, x_traces, filename, logscale=True):
    fig = plt.figure(figsize=(5, 4))
    f_traces = [f(x) for x in x_traces]

    if logscale:
        plt.semilogy(f_traces)
    else:
        plt.plot(f_traces)
    plt.xlabel('iteration (k)')
    plt.rc('font', family='serif')
    plt.ylabel('gap $f(x_k) - f(x^*)$')
    plt.tight_layout()

    fig.savefig(filename)
