import numpy as np


def proj_gd(fp, proj, x0, stepsize, tol=1e-5, maxiter=100000):
    """
    projected gradient descent for minimizing f(x) over X

    fp: function that takes an input x and returns the derivative of f at x
    proj: projection operator takes x as input and outputs its projection onto X 
    x0: initial point
    stepsize: constant step size used in gradient descent
    tol: toleracne parameter in the stopping crieterion. 
         Projected gradient descent stops when ||x_{k+1} - x_k|| < t * tol
    maxiter: maximum number of iterations in gradient descent.

    This function should return the sequence of approximate solutions x_k 
    produced by each iteration, and also the sequence y_k before projection
    """
    x_traces = [np.array(x0)]
    y_traces = []
    x = np.array(x0)
    for _ in range(maxiter):
        y = x - stepsize * fp(x)
        xp = proj(y)
        if np.linalg.norm(xp - x) < stepsize * tol:
            break
        x = xp
        y_traces.append(y)  # y is output of gradient step before projection
        x_traces.append(np.array(x))  # x is output of projected gradient step
    return x_traces, y_traces
