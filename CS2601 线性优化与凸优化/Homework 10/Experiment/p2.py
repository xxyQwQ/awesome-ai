import numpy as np
import LP

# parameter
c = np.array([8, 3, 0, 0], dtype=float)
A = np.array([[-1, -1, 1, 0], [-2, 1, 0, 1]], dtype=float)
b = np.array([-3, -1], dtype=float)

# solution
mu0 = np.array([4, 1, 2, 6], dtype=float)
mu_traces = LP.barrier(c, A, b, mu0)
for k, mu in enumerate(mu_traces):
    print('iteration %d: %s' % (k, mu))
print('dual optimal value:', -c @ mu_traces[-1])
