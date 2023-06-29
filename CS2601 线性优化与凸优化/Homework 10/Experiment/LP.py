import newton as nt
import numpy as np


def centering_step(c, A, b, x0, t):
    """
    c, A, b: parameters in LP

            min   c^T x
            s.t.  Ax = b
                  x >= 0

    x0: feasible initial point for constrained Newton's method
    t:  penalty parameter in barrier method

    This function returns the central point x^*(t) 
    """
    def F(x): return (c @ x - np.sum(np.log(np.maximum(x, 0))) / t)
    def Fp(x): return (c - (1 / x) / t)
    def Fpp(x): return (np.diag(1 / (x ** 2)) / t)
    x_traces = nt.newton_eq(F, Fp, Fpp, x0, A, b)
    return x_traces[-1]


def barrier(c, A, b, x0, tol=1e-8, t0=1, rho=10):
    """
    c, A, b: parameters in LP

            min   c^T x
            s.t.  Ax = b
                  x >= 0

    x0:  feasible initial point for the barrier method
    tol: tolerance parameter for the suboptimality gap. The algorithm stops when

             f(x) - f^* <= tol

    t0:  initial penalty parameter in barrier method
    rho: factor by which the penalty parameter is increased in each centering step

    This function should return a list of the iterates
    """
    t = t0
    x = np.array(x0)
    x_traces = [np.array(x0)]
    lim = len(b) / tol
    while t < lim:
        x = centering_step(c, A, b, x, t)
        x_traces.append(np.array(x))
        t = rho * t
    return x_traces
