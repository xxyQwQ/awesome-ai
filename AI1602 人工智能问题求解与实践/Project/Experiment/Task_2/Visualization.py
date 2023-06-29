import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('Agg')

data = np.load(r"../data/data2.npy")
plt.scatter(data[:, 0], data[:, 1], s=8, alpha=0.8)
plt.axis('equal')
plt.savefig('Distribution_Point.jpg')
plt.close()

plt.hist2d(data[:, 0], data[:, 1], bins=50)
plt.savefig('Distribution_Frequency.jpg')
plt.close()
