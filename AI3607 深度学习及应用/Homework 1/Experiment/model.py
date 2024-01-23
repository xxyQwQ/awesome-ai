from jittor import nn, Module


class RegressionModel(Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.layer_1 = nn.Linear(1, 32)
        self.layer_2 = nn.Linear(32, 32)
        self.layer_3 = nn.Linear(32, 32)
        self.layer_4 = nn.Linear(32, 1)


    def execute(self, x):
        x = nn.relu(self.layer_1(x))
        x = nn.relu(self.layer_2(x))
        x = nn.relu(self.layer_3(x))
        x = self.layer_4(x)
        return x
