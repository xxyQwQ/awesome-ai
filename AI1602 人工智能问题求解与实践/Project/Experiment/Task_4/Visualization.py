import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Algorithm import PCA

data = np.load(r"../data/data4.npy")
dataset = data.reshape(-1, 128)
model = PCA.PCA(dataset, component=3)
feature = model.result

plt.scatter(feature[:, 1], feature[:, 2], s=8, alpha=0.8)
plt.axis('equal')
plt.savefig('Distribution_3D_Front.jpg')
plt.close()

plt.scatter(feature[:, 0], feature[:, 2], s=8, alpha=0.8)
plt.axis('equal')
plt.savefig('Distribution_3D_Left.jpg')
plt.close()

plt.scatter(feature[:, 0], feature[:, 1], s=8, alpha=0.8)
plt.axis('equal')
plt.savefig('Distribution_3D_Above.jpg')
plt.close()

fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
ax.scatter(feature[:, 0], feature[:, 1], feature[:, 2], s=8, alpha=0.8)
plt.savefig("Distribution_3D.jpg")
plt.close()

plt.hist(feature[:, 2], bins=50, edgecolor="black")
plt.savefig('Distribution_Vertical.jpg')
plt.close()
