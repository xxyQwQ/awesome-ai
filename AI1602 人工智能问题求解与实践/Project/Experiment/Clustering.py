import numpy as np
from Algorithm import KMeans


def cluster_task_1(sample):
    dataset = sample.reshape(-1, 1)
    model = KMeans.KMeans(dataset, cluster=2, cycle=10)
    label = model.label
    print("Task 1 finished, return {} with {} length".format(type(label), len(label)))
    return label


def cluster_task_2(sample):
    dataset = sample.reshape(-1, 2)
    model = KMeans.KMeans(dataset, cluster=5, cycle=10)
    label = model.label
    print("Task 2 finished, return {} with {} length".format(type(label), len(label)))
    return label


def cluster_task_3(sample):
    dataset = sample.reshape(-1, 128)
    model = KMeans.KMeans(dataset, cluster=3, cycle=10)
    label = model.label
    print("Task 3 finished, return {} with {} length".format(type(label), len(label)))
    return label


def cluster_task_4(sample):
    dataset = sample.reshape(-1, 128)
    model = KMeans.KMeans(dataset, cluster=5, cycle=10, limit=25)
    label = model.label
    print("Task 4 finished, return {} with {} length".format(type(label), len(label)))
    return label


def main():
    data1 = np.load(r"data/data1.npy")
    data2 = np.load(r"data/data2.npy")
    data3 = np.load(r"data/data3.npy")
    data4 = np.load(r"data/data4.npy")

    label1 = cluster_task_1(data1)
    label2 = cluster_task_2(data2)
    label3 = cluster_task_3(data3)
    label4 = cluster_task_4(data4)

    np.save(r"output/label1.npy", label1)
    np.save(r"output/label2.npy", label2)
    np.save(r"output/label3.npy", label3)
    np.save(r"output/label4.npy", label4)


if __name__ == '__main__':
    main()
