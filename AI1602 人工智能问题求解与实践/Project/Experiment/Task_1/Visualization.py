import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('Agg')

data = np.load(r"../data/data1.npy")
plt.hist(data, bins=25, edgecolor="black")
plt.savefig('Distribution.jpg')
plt.close()
