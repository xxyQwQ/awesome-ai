import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Algorithm import PCA, KMeans

mpl.use('Agg')

data = np.load(r"../data/data4.npy")
dataset = data.reshape(-1, 128)
model = PCA.PCA(dataset, component=3)
feature = model.result
model = KMeans.KMeans(dataset, cluster=5, cycle=10, limit=25)
label = model.label

plt.scatter(feature[:, 1], feature[:, 2], c=label, s=8, alpha=0.8)
plt.axis('equal')
plt.savefig('Result_3D_Front.jpg')
plt.close()

plt.scatter(feature[:, 0], feature[:, 2], c=label, s=8, alpha=0.8)
plt.axis('equal')
plt.savefig('Result_3D_Left.jpg')
plt.close()

plt.scatter(feature[:, 0], feature[:, 1], c=label, s=8, alpha=0.8)
plt.axis('equal')
plt.savefig('Result_3D_Above.jpg')
plt.close()

fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
ax.scatter(feature[:, 0], feature[:, 1], feature[:, 2], c=label, s=8, alpha=0.8)
plt.savefig("Result_3D.jpg")
plt.close()
