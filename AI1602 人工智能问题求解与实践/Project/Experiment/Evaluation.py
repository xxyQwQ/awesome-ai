import numpy as np
from sklearn import metrics, cluster
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def score_task_2(data):
    print("---------- Task 2 ----------")
    dataset = data.reshape(-1, 2)
    order = []
    result = []
    for count in range(2, 11):
        label = cluster.KMeans(n_clusters=count).fit_predict(dataset)
        score = metrics.calinski_harabasz_score(dataset, label)
        print("Score of {} clusters is: {}".format(count, score))
        order.append(count)
        result.append(score)
    plt.plot(order, result)
    plt.savefig('figure/score2.jpg')
    plt.close()
    print("---------- Task 2 ----------\n")


def score_task_4(data):
    print("---------- Task 4 ----------")
    dataset = PCA(n_components=2).fit_transform(data.reshape(-1, 128))
    order = []
    result = []
    for count in range(2, 11):
        label = cluster.KMeans(n_clusters=count).fit_predict(dataset)
        score = metrics.calinski_harabasz_score(dataset, label)
        print("Score of {} clusters is: {}".format(count, score))
        order.append(count)
        result.append(score)
    plt.plot(order, result)
    plt.savefig('figure/score4.jpg')
    plt.close()
    print("---------- Task 4 ----------\n")


def main():
    data2 = np.load(r"data/data2.npy")
    data4 = np.load(r"data/data4.npy")
    score_task_2(data2)
    score_task_4(data4)


if __name__ == "__main__":
    main()
