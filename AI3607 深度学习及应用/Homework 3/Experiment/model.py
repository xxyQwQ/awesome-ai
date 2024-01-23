from jittor import nn, Module


class MLPModel(Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def execute(self, x):
        return self.network(x)


class RNNModel(Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(28, 256, 1, batch_first=True)
        self.fc = nn.Linear(256, 10)

    def execute(self, x):
        x, _ = self.rnn(x.view(-1, 28, 28))
        x = self.fc(x[-1, :, :])
        return x


class LSTMModel(Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(28, 256, 1, batch_first=True)
        self.fc = nn.Linear(256, 10)

    def execute(self, x):
        x, _ = self.lstm(x.view(-1, 28, 28))
        x = self.fc(x[-1, :, :])
        return x
