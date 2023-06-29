import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

data = np.load('data_example.npy')
label = np.load('label_example.npy')

plt.scatter(data[:,0], data[:,1], c=label)
plt.axis('equal')
plt.savefig('data_example.jpg')

print([(data[i], label[i]) for i in range(10)])
