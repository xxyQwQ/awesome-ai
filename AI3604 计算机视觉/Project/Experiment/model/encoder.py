from torch import nn
from torchvision.models import resnet18

from model.block import DownsampleBlock, UpsampleBlock


class StructureEncoder(nn.Module):
    def __init__(self):
        super(StructureEncoder, self).__init__()

        self.downsample_1 = DownsampleBlock(1, 32)
        self.downsample_2 = DownsampleBlock(32, 64)
        self.downsample_3 = DownsampleBlock(64, 128)
        self.downsample_4 = DownsampleBlock(128, 256)
        self.downsample_5 = DownsampleBlock(256, 512)
        self.downsample_6 = DownsampleBlock(512, 512)

        self.upsample_1 = UpsampleBlock(512, 512)
        self.upsample_2 = UpsampleBlock(1024, 256)
        self.upsample_3 = UpsampleBlock(512, 128)
        self.upsample_4 = UpsampleBlock(256, 64)
        self.upsample_5 = UpsampleBlock(128, 32)
        self.upsample_6 = UpsampleBlock(64, 64)

    def forward(self, input):                                   # (batch_size, 1, 128, 128)
        feature_1 = self.downsample_1(input)                    # (batch_size, 32, 64, 64)
        feature_2 = self.downsample_2(feature_1)                # (batch_size, 64, 32, 32)
        feature_3 = self.downsample_3(feature_2)                # (batch_size, 128, 16, 16)
        feature_4 = self.downsample_4(feature_3)                # (batch_size, 256, 8, 8)
        feature_5 = self.downsample_5(feature_4)                # (batch_size, 512, 4, 4)
        attribute_1 = self.downsample_6(feature_5)              # (batch_size, 512, 2, 2)

        attribute_2 = self.upsample_1(attribute_1, feature_5)   # (batch_size, 1024, 4, 4)
        attribute_3 = self.upsample_2(attribute_2, feature_4)   # (batch_size, 512, 8, 8)
        attribute_4 = self.upsample_3(attribute_3, feature_3)   # (batch_size, 256, 16, 16)
        attribute_5 = self.upsample_4(attribute_4, feature_2)   # (batch_size, 128, 32, 32)
        attribute_6 = self.upsample_5(attribute_5, feature_1)   # (batch_size, 64, 64, 64)
        attribute_7 = self.upsample_6(attribute_6)              # (batch_size, 64, 128, 128)

        return attribute_1, attribute_2, attribute_3, attribute_4, attribute_5, attribute_6, attribute_7


class StyleEncoder(nn.Module):
    def __init__(self, reference_count: int):
        super(StyleEncoder, self).__init__()
        self.backbone = resnet18()
        self.backbone.conv1 = nn.Conv2d(reference_count, 64, 7, 2, 3)
        self.backbone.fc = nn.Linear(512, 512)

    def forward(self, input):
        output = self.backbone(input)   # (batch_size, 512)
        return output
