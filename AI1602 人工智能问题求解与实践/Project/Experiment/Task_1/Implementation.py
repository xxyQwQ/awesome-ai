import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from Algorithm import KMeans

mpl.use('Agg')

data = np.load(r"../data/data1.npy")
dataset = data.reshape(-1, 1)
y = np.zeros(1000)
model = KMeans.KMeans(dataset, cluster=2, cycle=10)
label = model.label

plt.scatter(data, y, c=label, s=8, alpha=0.8)
plt.axis('equal')
plt.savefig('Result.jpg')
plt.close()
