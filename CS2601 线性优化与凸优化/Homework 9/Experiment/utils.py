import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp


def plot_traces_2d(f_2d, x_traces, y_traces, feasible_set=None, filename=None):
    fig = plt.figure(figsize=(3.5, 2.5))
    fig, ax = plt.subplots()
    ax.set_aspect("equal")

    plt.plot(*zip(*x_traces), '-o', color='red')

    for k in range(len(y_traces)):
        x = x_traces[k]
        y = y_traces[k]
        xp = x_traces[k+1]
        plt.arrow(*x, *(y-x), color='green', length_includes_head=True,
                  head_length=0.05, head_width=0.02)
        plt.arrow(*y, *(xp-y), color='magenta', length_includes_head=True,
                  head_length=0.05, head_width=0.02)

    x1, x2 = zip(*np.append(x_traces, y_traces, axis=0))
    w = max(max(x1) - min(x1), max(x2) - min(x2))/2 + 0.2
    x1 = np.arange(*(min(x1)/2 + max(x1)/2 + [-w, w]), 0.01)
    x2 = np.arange(*(min(x2)/2 + max(x2)/2 + [-w, w]), 0.01)

    x1, x2 = np.meshgrid(x1, x2)
    plt.contour(x1, x2, f_2d(x1, x2), 20, colors='blue')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.tight_layout(pad=.1)

    if feasible_set is not None:
        ax.add_patch(feasible_set)

    if filename is not None:
        fig.savefig(filename)

    return fig


def plot(f, x_traces, filename=None, color='blue', logscale=True):
    fig = plt.figure(figsize=(3.5, 2.5))
    f_traces = [f(x) for x in x_traces]

    if logscale:
        plt.semilogy(f_traces, color=color)
    else:
        plt.plot(f_traces, color=color)
    plt.xlabel('iteration (k)')
    plt.rc('text', usetex=False)
    plt.rc('font', family='serif')
    plt.ylabel('gap $f(x_k) - f(x^*)$')
    plt.tight_layout(pad=.1)

    if filename is not None:
        fig.savefig(filename)

    return fig
