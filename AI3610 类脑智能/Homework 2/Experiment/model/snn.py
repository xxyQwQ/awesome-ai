from torch import nn
from spikingjelly.activation_based import layer, neuron


class MNIST_SNN(nn.Module):
    def __init__(self, spiking_neuron='LIF'):
        super().__init__()

        if spiking_neuron == 'IF':
            spiking_neuron = neuron.IFNode
        elif spiking_neuron == 'LIF':
            spiking_neuron = neuron.LIFNode
        
        self.conv_layers = nn.Sequential(
            layer.Conv2d(2, 16, 3, padding=0, bias=False),
            layer.BatchNorm2d(16),
            spiking_neuron(),
            layer.MaxPool2d(2, 2),
            # (time_steps, batch_size, 16, 16, 16)
            layer.Conv2d(16, 32, 3, padding=1, bias=False),
            layer.BatchNorm2d(32),
            spiking_neuron(),
            layer.MaxPool2d(2, 2),
            # (time_steps, batch_size, 32, 8, 8)
            layer.Conv2d(32, 64, 3, padding=1, bias=False),
            layer.BatchNorm2d(64),
            spiking_neuron(),
            layer.MaxPool2d(2, 2)
            # (time_steps, batch_size, 64, 4, 4)
        )

        self.fc_layers = nn.Sequential(
            layer.Flatten(),
            # (time_steps, batch_size, 1024)
            layer.Linear(1024, 256),
            spiking_neuron(),
            # (time_steps, batch_size, 256)
            layer.Linear(256, 10),
            spiking_neuron()
            # (time_steps, batch_size, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
