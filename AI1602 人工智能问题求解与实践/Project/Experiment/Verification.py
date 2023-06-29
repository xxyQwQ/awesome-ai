import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Algorithm import PCA


def plot_task_1(data, label):
    plt.scatter(data, np.zeros(1000), c=label, s=8, alpha=0.8)
    plt.axis('equal')
    plt.savefig('figure/figure1.jpg')
    plt.close()


def plot_task_2(data, label):
    plt.scatter(data[:, 0], data[:, 1], c=label, s=8, alpha=0.8)
    plt.axis('equal')
    plt.savefig('figure/figure2.jpg')
    plt.close()


def plot_task_3(data, label):
    dataset = data.reshape(-1, 128)
    model = PCA.PCA(dataset, component=2)
    feature = model.result
    plt.scatter(feature[:, 0], feature[:, 1], c=label, s=8, alpha=0.8)
    plt.axis('equal')
    plt.savefig('figure/figure3.jpg')
    plt.close()


def plot_task_4(data, label):
    dataset = data.reshape(-1, 128)
    model = PCA.PCA(dataset, component=3)
    feature = model.result

    plt.scatter(feature[:, 1], feature[:, 2], c=label, s=8, alpha=0.8)
    plt.axis('equal')
    plt.savefig('figure/figure4_1.jpg')
    plt.close()

    plt.scatter(feature[:, 0], feature[:, 2], c=label, s=8, alpha=0.8)
    plt.axis('equal')
    plt.savefig('figure/figure4_2.jpg')
    plt.close()

    plt.scatter(feature[:, 0], feature[:, 1], c=label, s=8, alpha=0.8)
    plt.axis('equal')
    plt.savefig('figure/figure4_3.jpg')
    plt.close()

    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.scatter(feature[:, 0], feature[:, 1], feature[:, 2], c=label, s=8, alpha=0.8)
    plt.savefig("figure/figure4_0.jpg")
    plt.close()


def main():
    data1 = np.load(r"data/data1.npy")
    data2 = np.load(r"data/data2.npy")
    data3 = np.load(r"data/data3.npy")
    data4 = np.load(r"data/data4.npy")
    label1 = np.load(r"output/label1.npy")
    label2 = np.load(r"output/label2.npy")
    label3 = np.load(r"output/label3.npy")
    label4 = np.load(r"output/label4.npy")
    plot_task_1(data1, label1)
    plot_task_2(data2, label2)
    plot_task_3(data3, label3)
    plot_task_4(data4, label4)


if __name__ == "__main__":
    main()
