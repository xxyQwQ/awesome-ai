from torch import nn
import torch.nn.functional as F


def linear_softmax_pooling(x):
    return (x ** 2).sum(1) / x.sum(1)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pooling=None):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(pooling) if pooling else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.pool:
            x = self.pool(x)
        return x


class CRNN(nn.Module):
    def __init__(self, num_bins, num_classes, **kwargs):
        super(CRNN, self).__init__()
        self.num_bins = num_bins
        self.num_classes = num_classes
        self.num_blocks = kwargs['num_blocks']
        self.num_layers = kwargs['num_layers']
        self.hidden_dims = kwargs['hidden_dims']
        self.bn = nn.BatchNorm1d(self.num_bins)
        if self.num_blocks == 1:
            block_list = [(1, self.hidden_dims, (2, 2))]
        elif self.num_blocks == 2:
            block_list = [(1, 16, (2, 2)), (16, self.hidden_dims, (2, 2))]
        elif self.num_blocks == 3:
            block_list = [(1, 16, (2, 2)), (16, 32, (2, 2)), (32, self.hidden_dims, (1, 2))]
        elif self.num_blocks == 4:
            block_list = [(1, 16, (2, 2)), (16, 32, (2, 2)), (32, 64, (1, 2)), (64, self.hidden_dims, (1, 2))]
        elif self.num_blocks == 5:
            block_list = [(1, 16, (2, 2)), (16, 32, (2, 2)), (32, 64, (1, 2)), (64, 128, (1, 2)), (128, self.hidden_dims, (1, 2))]
        self.cb = nn.Sequential(*[ConvBlock(in_channels, out_channels, pooling=pooling) for in_channels, out_channels, pooling in block_list])
        self.gru = nn.GRU(self.hidden_dims, self.hidden_dims, num_layers=self.num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * self.hidden_dims, self.num_classes)
        self.sigmoid = nn.Sigmoid()

    def detection(self, x):
        time_steps = x.shape[1]
        x = x.transpose(1, 2) # (batch_size, num_bins, time_steps)
        x = self.bn(x) # (batch_size, num_bins, time_steps)
        x = x.transpose(1, 2) # (batch_size, time_steps, num_bins)
        x = x.unsqueeze(1) # (batch_size, 1, time_steps, num_bins)
        x = self.cb(x) # (batch_size, hidden_dims, sequence_length, feature_dims)
        x = x.mean(3) # (batch_size, hidden_dims, sequence_length)
        x = x.transpose(1, 2) # (batch_size, sequence_length, hidden_dims)
        x, _ = self.gru(x) # (batch_size, sequence_length, 2 * hidden_dims)
        x = self.fc(x) # (batch_size, sequence_length, num_classes)
        x = self.sigmoid(x) # (batch_size, sequence_length, num_classes)
        x = x.transpose(1, 2) # (batch_size, num_classes, sequence_length)
        x = F.interpolate(x, size=time_steps, mode='linear') # (batch_size, num_classes, time_steps)
        x = x.transpose(1, 2) # (batch_size, time_steps, num_classes)
        return x

    def forward(self, x):
        frame_prob = self.detection(x) # (batch_size, time_steps, num_classes)
        clip_prob = linear_softmax_pooling(frame_prob) # (batch_size, num_classes)
        return {'clip_prob': clip_prob, 'frame_prob': frame_prob}
