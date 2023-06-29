import numpy as np


class PCA:
    def __init__(self, data, component=1):
        self.data = data
        self.component = component
        self.total, self.dimension = data.shape
        if self.component >= self.total or self.component >= self.dimension:
            raise "Invalid component"
        self.transfer = None
        self.result = None
        self.train()

    def train(self):
        mean = np.mean(self.data, axis=0)
        cov = np.cov(self.data - mean, rowvar=False)
        value, vector = np.linalg.eig(cov)
        order = np.argsort(value)
        target = order[-self.component:]
        self.transfer = np.real(vector[:, target])
        self.result = np.dot(self.data - mean, self.transfer)


if __name__ == "__main__":
    dataset = np.random.random((10, 3))
    model = PCA(dataset, component=1)
    result = model.result
    print(dataset)
    print(result)
