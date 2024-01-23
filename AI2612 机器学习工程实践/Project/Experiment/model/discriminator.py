from torch import nn


class FundamentalDiscriminator(nn.Module):
    def __init__(self, image_size: int):
        super(FundamentalDiscriminator, self).__init__()
        block_list = []

        # embedding
        num_channels = 64
        image_size = image_size // 2
        block_list.append([
            nn.Conv2d(3, num_channels, 4, 2, 1),
            nn.InstanceNorm2d(num_channels),
            nn.LeakyReLU(0.2, inplace=True)
        ])

        # downsample
        while image_size > 8:
            pre_channels = num_channels
            num_channels = min(512, 2 * num_channels)
            image_size = image_size // 2
            block_list.append([
                nn.Conv2d(pre_channels, num_channels, 4, 2, 1),
                nn.InstanceNorm2d(num_channels),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        
        # classify
        block_list.append([
            nn.Conv2d(num_channels, num_channels, 4, 1, 1),
            nn.InstanceNorm2d(num_channels),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        block_list.append([nn.Conv2d(num_channels, 1, 4, 1, 1)])

        layer_list = [layer for block in block_list for layer in block]
        self.network = nn.Sequential(*layer_list)
        self.layer_list = layer_list
    
    def forward(self, input):
        output = self.network(input)
        return output


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, num_scales: int = 3, image_size: int = 224):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_scales = num_scales

        network_list = []
        for _ in range(self.num_scales):
            network_list.append(FundamentalDiscriminator(image_size))
            image_size = image_size // 2
        self.network = nn.ModuleList(network_list)

        self.downsample = nn.AvgPool2d(3, 2, 1, count_include_pad=False)
    
    def forward(self, input):
        output = []
        for i in range(self.num_scales):
            output.append(self.network[i](input))
            if i < self.num_scales - 1:
                input = self.downsample(input)
        return output
