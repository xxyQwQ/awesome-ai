import numpy as np


def soft_th(w, th):
    """
    w: input vector
    th: threshold

        soft_th(w, th) = sgn(w) * (|w| - th)^+

    """
    return np.sign(w) * np.maximum(np.abs(w) - th, 0)


def ista(X, y, lambd, w0, stepsize, tol=1e-9, maxiter=100000):
    """
    This function implements the iterative soft-thresholding algorithm (ISTA) for
    solving Lasso in penalized form

        minimize 0.5 * ||X * w - y||_2^2 + lambd * ||w||_1

    w0: initial point
    stepsize: constant step size used in ISTA
    tol: toleracne parameter in the stopping criterion. Here we stop when the
         norm of the difference between two succesive iterates is smaller than tol
    maxiter: maximum number of iterations in ISTA.
    """
    w_traces = [np.array(w0)]
    w = np.array(w0)
    for _ in range(maxiter):
        grad = X.T @ (X @ w - y)
        w_next = soft_th(w - stepsize * grad, lambd * stepsize)
        if np.linalg.norm(w_next - w) < tol:
            break
        w = w_next
        w_traces.append(np.array(w))
    return w_traces
