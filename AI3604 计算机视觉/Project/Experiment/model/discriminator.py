import torch
from torch import nn

from model.block import DownsampleBlock


class EnhancedDiscriminator(nn.Module):
    def __init__(self, writer_count: int, character_count: int, image_size: int):
        super(EnhancedDiscriminator, self).__init__()

        self.feature_extractor = nn.Sequential(
            DownsampleBlock(1, 64),
            DownsampleBlock(64, 128),
            DownsampleBlock(128, 256),
            DownsampleBlock(256, 512),
            DownsampleBlock(512, 512),
            nn.Flatten(),
        )

        feature_size = 512 * (image_size // 32) ** 2
        self.reality_classifier = nn.Linear(feature_size, 1)
        self.writer_classifier = nn.Linear(feature_size, writer_count)
        self.character_classifier = nn.Linear(feature_size, character_count)

    def forward(self, input):
        feature = self.feature_extractor(input)

        reality = torch.sigmoid(self.reality_classifier(feature))
        writer = self.writer_classifier(feature)
        character = self.character_classifier(feature)

        return reality, writer, character


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, writer_count: int, character_count: int, num_scales: int = 3, image_size: int = 128):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_scales = num_scales

        network_list = []
        for _ in range(num_scales):
            network_list.append(EnhancedDiscriminator(writer_count, character_count, image_size))
            image_size //= 2
        self.network = nn.ModuleList(network_list)

        self.downsample = nn.AvgPool2d(3, 2, 1, count_include_pad=False)

    def forward(self, input):
        output = []
        for i in range(self.num_scales):
            output.append(self.network[i](input))
            input = self.downsample(input)
        return output
