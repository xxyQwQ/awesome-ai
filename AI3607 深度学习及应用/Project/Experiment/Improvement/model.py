from jittor import nn, Module
import pygmtools
from pygmtools import sinkhorn
pygmtools.BACKEND = 'jittor'


class NaiveCNN(Module):
    def __init__(self, segment=2):
        super(NaiveCNN, self).__init__()
        self.segment, self.block = segment, segment**2
        self.feature = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )

    def execute(self, x):
        return self.feature(x)


class NaiveFCN(Module):
    def __init__(self, segment=2):
        super(NaiveFCN, self).__init__()
        self.segment, self.block = segment, segment**2
        self.network = nn.Sequential(
            nn.Linear(1024*self.block, 4096),
            nn.BatchNorm(4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.BatchNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, self.block**2),
        )

    def execute(self, x):
        return self.network(x)


class PuzzleSolver(Module):
    def __init__(self, segment=2):
        super(PuzzleSolver, self).__init__()
        self.segment, self.block = segment, segment**2
        self.extractor = NaiveCNN(segment=self.segment)
        self.aggregator = NaiveFCN(segment=self.segment)

    def execute(self, x):
        shape = x.shape
        x = x.view(-1, *shape[2:])
        x = self.extractor(x)
        x = x.view(shape[0], -1)
        x = self.aggregator(x)
        x = x.view(-1, self.block, self.block)
        return x if self.is_training else sinkhorn(x)
