from jittor import nn, Module
import pygmtools
from pygmtools import sinkhorn
pygmtools.BACKEND = 'jittor'


class Extractor(Module):
    def __init__(self, block=4):
        super(Extractor, self).__init__()
        if block == 4:
            self.network = nn.Sequential(
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
                nn.Flatten(),
                nn.Linear(1024, 512),
                nn.ReLU(),
            )
        elif block == 9:
            self.network = nn.Sequential(
                nn.Conv2d(3, 32, 3, 1, 2),
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
                nn.Flatten(),
                nn.Linear(576, 512),
                nn.ReLU(),
            )

    def execute(self, x):
        return self.network(x)


class Aggregator(Module):
    def __init__(self, block=4):
        super(Aggregator, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(block*512, 4096),
            nn.ReLU(),
            nn.Linear(4096, block**2),
        )

    def execute(self, x):
        return self.network(x)


class PuzzleSolver(Module):
    def __init__(self, block=4):
        super(PuzzleSolver, self).__init__()
        self.block, self.type = block, type
        self.extractor = Extractor(block=self.block)
        self.aggregator = Aggregator(block=self.block)

    def execute(self, x):
        shape = x.shape
        x = x.view(-1, *shape[2:])
        x = self.extractor(x)
        x = x.view(shape[0], -1)
        x = self.aggregator(x)
        x = x.view(-1, self.block, self.block)
        return sinkhorn(x)
