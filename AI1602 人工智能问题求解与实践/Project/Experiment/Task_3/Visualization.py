import numpy as np
import matplotlib.pyplot as plt
from Algorithm import PCA

data = np.load(r"../data/data3.npy")
dataset = data.reshape(-1, 128)
model = PCA.PCA(dataset, component=2)
feature = model.result

plt.scatter(feature[:, 0], feature[:, 1], s=8, alpha=0.8)
plt.axis('equal')
plt.savefig('Distribution.jpg')
plt.close()
