import torch
from torch import nn


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(DownsampleBlock, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, input):
        output = self.network(input)
        return output


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(UpsampleBlock, self).__init__()
        self.network = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, input, injection=None):
        output = self.network(input)
        if injection is not None:
            output = torch.cat((output, injection), dim=1)
        return output


class AttributeEncoder(nn.Module):
    def __init__(self):
        super(AttributeEncoder, self).__init__()

        self.downsample_1 = DownsampleBlock(3, 32, 4, 2, 1)
        self.downsample_2 = DownsampleBlock(32, 64, 4, 2, 1)
        self.downsample_3 = DownsampleBlock(64, 128, 4, 2, 1)
        self.downsample_4 = DownsampleBlock(128, 256, 4, 2, 1)
        self.downsample_5 = DownsampleBlock(256, 512, 4, 2, 2)
        self.downsample_6 = DownsampleBlock(512, 1024, 4, 2, 1)
        self.downsample_7 = DownsampleBlock(1024, 1024, 4, 2, 1)

        self.upsample_1 = UpsampleBlock(1024, 1024, 4, 2, 1)
        self.upsample_2 = UpsampleBlock(2048, 512, 4, 2, 1)
        self.upsample_3 = UpsampleBlock(1024, 256, 4, 2, 2)
        self.upsample_4 = UpsampleBlock(512, 128, 4, 2, 1)
        self.upsample_5 = UpsampleBlock(256, 64, 4, 2, 1)
        self.upsample_6 = UpsampleBlock(128, 32, 4, 2, 1)
        self.upsample_7 = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, input):                                   # (batch_size, 3, 224, 224)
        feature_1 = self.downsample_1(input)                    # (batch_size, 32, 112, 112)
        feature_2 = self.downsample_2(feature_1)                # (batch_size, 64, 56, 56)
        feature_3 = self.downsample_3(feature_2)                # (batch_size, 128, 28, 28)
        feature_4 = self.downsample_4(feature_3)                # (batch_size, 256, 14, 14)
        feature_5 = self.downsample_5(feature_4)                # (batch_size, 512, 8, 8)
        feature_6 = self.downsample_6(feature_5)                # (batch_size, 1024, 4, 4)
        attribute_1 = self.downsample_7(feature_6)              # (batch_size, 1024, 2, 2)

        attribute_2 = self.upsample_1(attribute_1, feature_6)   # (batch_size, 2048, 4, 4)
        attribute_3 = self.upsample_2(attribute_2, feature_5)   # (batch_size, 1024, 8, 8)
        attribute_4 = self.upsample_3(attribute_3, feature_4)   # (batch_size, 512, 14, 14)
        attribute_5 = self.upsample_4(attribute_4, feature_3)   # (batch_size, 256, 28, 28)
        attribute_6 = self.upsample_5(attribute_5, feature_2)   # (batch_size, 128, 56, 56)
        attribute_7 = self.upsample_6(attribute_6, feature_1)   # (batch_size, 64, 112, 112)
        attribute_8 = self.upsample_7(attribute_7)              # (batch_size, 64, 224, 224)

        return attribute_1, attribute_2, attribute_3, attribute_4, attribute_5, attribute_6, attribute_7, attribute_8
