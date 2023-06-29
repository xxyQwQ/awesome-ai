import numpy as np


def newton(fp, fpp, x0, tol=1e-5, maxiter=100000):
    """
    fp: function that takes an input x and returns the gradient of f at x
    fpp: function that takes an input x and returns the Hessian of f at x
    x0: initial point  
    tol: toleracne parameter in the stopping criterion. Newton's method stops 
         when the 2-norm of the gradient is smaller than tol
    maxiter: maximum number of iterations

    This function should return a list of the sequence of approximate solutions
    x_k produced by each iteration
    """
    x_traces = [np.array(x0)]
    x = np.array(x0)
    for _ in range(maxiter):
        grad = fp(x)
        if np.linalg.norm(grad) < tol:
            break
        x = x - np.linalg.solve(fpp(x), grad)
        x_traces.append(np.array(x))
    return x_traces


def damped_newton(f, fp, fpp, x0, alpha=0.5, beta=0.5, tol=1e-5, maxiter=100000):
    """
    f: function that takes an input x an returns the value of f at x
    fp: function that takes an input x and returns the gradient of f at x
    fpp: function that takes an input x and returns the Hessian of f at x
    x0: initial point in gradient descent
    alpha: parameter in Armijo's rule 
                            f(x + t * d) > f(x) + t * alpha * <f'(x), d>
    beta: constant factor used in stepsize reduction
    tol: toleracne parameter in the stopping crieterion. Here we stop
         when the 2-norm of the gradient is smaller than tol
    maxiter: maximum number of iterations in gradient descent.

    This function should return a list of the sequence of approximate solutions
    x_k produced by each iteration and the total number of iterations in the inner loop
    """
    x_traces = [np.array(x0)]
    stepsize_traces = []
    tot_num_iter = 0
    x = np.array(x0)
    for _ in range(maxiter):
        grad = fp(x)
        if np.linalg.norm(grad) < tol:
            break
        direction = np.linalg.solve(fpp(x), grad)
        stepsize = 1.0
        while f(x - stepsize * direction) > f(x) + alpha * stepsize * (grad @ direction):
            stepsize = beta * stepsize
            tot_num_iter += 1
        x = x - stepsize * direction
        x_traces.append(np.array(x))
        stepsize_traces.append(stepsize)
    return x_traces, stepsize_traces, tot_num_iter


def newton_eq(f, fp, fpp, x0, A, b, initial_stepsize=1.0, alpha=0.5, beta=0.5, tol=1e-8, maxiter=100000):
    """
    f: function that takes an input x an returns the value of f at x
    fp: function that takes an input x and returns the gradient of f at x
    fpp: function that takes an input x and returns the Hessian of f at x
    A, b: constraint A x = b
    x0: initial feasible point
    initial_stepsize: initial stepsize used in backtracking line search
    alpha: parameter in Armijo's rule 
                            f(x + t * d) > f(x) + t * alpha * f(x) @ d
    beta: constant factor used in stepsize reduction
    tol: toleracne parameter in the stopping crieterion. Gradient descent stops 
         when the 2-norm of the Newton direction is smaller than tol
    maxiter: maximum number of iterations in outer loop of damped Newton's method.

    This function should return a list of the iterates x_k
    """
    x_traces = [np.array(x0)]
    m = len(b)
    x = np.array(x0)
    for _ in range(maxiter):
        gradient = fp(x)
        hessian = fpp(x)
        solution = np.linalg.solve(
            np.block([[hessian, A.T], [A, np.zeros((m, m))]]),
            np.block([[-1 * gradient.reshape(-1, 1)], [np.zeros((m, 1))]])
        )
        d = solution[0: len(x)].reshape(-1)
        if np.linalg.norm(d) < tol:
            break
        stepsize = initial_stepsize
        k = 1
        while(f(x + stepsize * d) > f(x) + alpha * stepsize * (gradient @ d)):
            stepsize = beta * stepsize
            k += 1
            if k > 5:
                break
        x = x + stepsize * d
        x_traces.append(np.array(x))
    return x_traces
