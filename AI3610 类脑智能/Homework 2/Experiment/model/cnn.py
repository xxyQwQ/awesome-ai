from torch import nn


class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # (batch_size, 16, 16, 16)
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # (batch_size, 32, 8, 8)
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
            # (batch_size, 64, 4, 4)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            # (batch_size, 1024)
            nn.Linear(1024, 256),
            nn.ReLU(),
            # (batch_size, 256)
            nn.Linear(256, 10),
            # (batch_size, 10)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
