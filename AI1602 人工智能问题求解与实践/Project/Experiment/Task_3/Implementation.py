import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from Algorithm import PCA, KMeans

mpl.use('Agg')

data = np.load(r"../data/data3.npy")
dataset = data.reshape(-1, 128)
model = PCA.PCA(dataset, component=2)
feature = model.result
model = KMeans.KMeans(dataset, cluster=3, cycle=10)
label = model.label

plt.scatter(feature[:, 0], feature[:, 1], c=label, s=8, alpha=0.8)
plt.axis('equal')
plt.savefig('Result.jpg')
plt.close()
