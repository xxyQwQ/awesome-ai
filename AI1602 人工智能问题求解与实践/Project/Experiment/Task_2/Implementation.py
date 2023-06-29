import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from Algorithm import KMeans

mpl.use('Agg')

data = np.load(r"../data/data2.npy")
dataset = data.reshape(-1, 2)
model = KMeans.KMeans(dataset, cluster=5, cycle=10)
label = model.label

plt.scatter(data[:, 0], data[:, 1], c=label, s=8, alpha=0.8)
plt.axis('equal')
plt.savefig('Result.jpg')
plt.close()
