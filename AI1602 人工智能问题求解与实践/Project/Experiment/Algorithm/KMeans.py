import numpy as np


class KMeans:
    def __init__(self, data, cluster=1, seed=0, limit=int(2**20), cycle=1):
        self.data = data
        self.cluster = cluster
        self.seed = seed
        np.random.seed(seed)
        self.limit = limit
        self.cycle = cycle
        self.total, self.dimension = data.shape
        self.epsilon = float(1e-4)
        self.label = np.zeros(self.total, np.int32)
        self.best = limit
        for i in range(self.cycle):
            self.single_train()

    def single_train(self):
        label = np.zeros(self.total, np.int32)
        center = self.data[np.random.randint(self.total, size=self.cluster)]
        iterations = 0
        converge = False
        while not converge:
            iterations += 1
            if iterations > self.limit:
                break
            for i in range(self.total):
                evaluation = np.sum(np.square(self.data[i] - center), axis=1)
                label[i] = np.argmin(evaluation)
            converge = True
            for i in range(self.cluster):
                group = self.data[label == i]
                if len(group) == 0:
                    continue
                update = np.mean(group, axis=0)
                delta = np.sum(np.abs(center[i] - update), axis=0)
                if delta > self.epsilon:
                    center[i] = update
                    converge = False
        if iterations < self.best:
            self.label = label
            self.best = iterations


if __name__ == "__main__":
    dataset = np.random.rand(10, 2)
    model = KMeans(dataset, cluster=4, limit=500, cycle=10)
    label = model.label
    print(dataset)
    print(label)
