import numpy as np
import matplotlib.pyplot as plt
import LP

# parameter
c = np.array([-3, -1, 0, 0], dtype=float)
A = np.array([[1, 2, 1, 0], [1, -1, 0, 1]], dtype=float)
b = np.array([8, 3], dtype=float)

# solution
x0 = np.array([2, 1, 4, 2], dtype=float)
x_traces = LP.barrier(c, A, b, x0)
for k, x in enumerate(x_traces):
    print('iteration {}: {}'.format(k, x))
print('optimal value: {}'.format(c @ x_traces[-1]))

# visualization
x12 = [x[0:2] for x in x_traces]
fig = plt.figure()
plt.plot([0, 3, 14/3, 0, 0], [0, 0, 5/3, 4, 0], color='blue', linewidth=3)
line, = plt.plot(*zip(*x12), color='red', linewidth=3,  label='barrier method')
plt.scatter(*zip(*x12), color='red', marker='o', s=50)
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(handles=[line], fontsize=20)
plt.tight_layout()
fig.savefig('./figures/p1.png')
