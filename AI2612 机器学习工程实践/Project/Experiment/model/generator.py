from torch import nn
from model.encoder import AttributeEncoder


class DenormalizationLayer(nn.Module):
    def __init__(self, input_channels: int, attribute_channels: int, identity_channels: int):
        super(DenormalizationLayer, self).__init__()
        self.norm = nn.BatchNorm2d(input_channels)

        self.conv_mu = nn.Conv2d(attribute_channels, input_channels, 1, 1, 0)
        self.conv_sigma = nn.Conv2d(attribute_channels, input_channels, 1, 1, 0)

        self.fc_mu = nn.Linear(identity_channels, input_channels)
        self.fc_sigma = nn.Linear(identity_channels, input_channels)

        self.conv_input = nn.Conv2d(input_channels, 1, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, attribute, identity):
        input = self.norm(input)

        attribute_mu = self.conv_mu(attribute)
        attribute_sigma = self.conv_sigma(attribute)
        attribute_activation = attribute_mu + attribute_sigma * input

        target_shape = (input.shape[0], input.shape[1], 1, 1)
        identity_mu = self.fc_mu(identity).reshape(target_shape).expand_as(input)
        identity_sigma = self.fc_sigma(identity).reshape(target_shape).expand_as(input)
        identity_activation = identity_mu + identity_sigma * input

        mask = self.sigmoid(self.conv_input(input))
        output = (1 - mask) * attribute_activation + mask * identity_activation
        return output


class DenormalizationBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, attribute_channels: int, identity_channels: int):
        super(DenormalizationBlock, self).__init__()
        self.transform = bool(input_channels != output_channels)
        self.relu = nn.ReLU(inplace=True)

        self.denorm_1 = DenormalizationLayer(input_channels, attribute_channels, identity_channels)
        self.conv_1 = nn.Conv2d(input_channels, input_channels, 3, 1, 1)

        self.denorm_2 = DenormalizationLayer(input_channels, attribute_channels, identity_channels)
        self.conv_2 = nn.Conv2d(input_channels, output_channels, 3, 1, 1)

        if self.transform:
            self.denorm_3 = DenormalizationLayer(input_channels, attribute_channels, identity_channels)
            self.conv_3 = nn.Conv2d(input_channels, output_channels, 3, 1, 1)
    
    def forward(self, input, attribute, identity):
        output = self.denorm_1(input, attribute, identity)
        output = self.conv_1(self.relu(output))

        output = self.denorm_2(output, attribute, identity)
        output = self.conv_2(self.relu(output))

        if self.transform:
            input = self.denorm_3(input, attribute, identity)
            input = self.conv_3(self.relu(input))

        output += input
        return output


class DenormalizationGenerator(nn.Module):
    def __init__(self, identity_channels: int):
        super(DenormalizationGenerator, self).__init__()
        self.num_blocks = 8

        self.boot = nn.ConvTranspose2d(identity_channels, 1024, 2, 1, 0)
        self.tanh = nn.Tanh()

        input_channels = [1024, 1024, 1024, 1024, 512, 256, 128, 64]
        output_channels = [1024, 1024, 1024, 512, 256, 128, 64, 3]
        attribute_channels = [1024, 2048, 1024, 512, 256, 128, 64, 64]
        upsample_targets = [4, 8, 14, 28, 56, 112, 224]

        denorm_list = []
        upsample_list = []
        for i in range(self.num_blocks):
            denorm_list.append(DenormalizationBlock(input_channels[i], output_channels[i], attribute_channels[i], identity_channels))
            if i < self.num_blocks - 1:
                upsample_list.append(nn.UpsamplingBilinear2d(size=upsample_targets[i]))
        self.denorm = nn.ModuleList(denorm_list)
        self.upsample = nn.ModuleList(upsample_list)

    def forward(self, attribute, identity):
        target_shape = (identity.shape[0], identity.shape[1], 1, 1)
        output = self.boot(identity.reshape(target_shape))
        for i in range(self.num_blocks):
            output = self.denorm[i](output, attribute[i], identity)
            if i < self.num_blocks - 1:
                output = self.upsample[i](output)
        output = self.tanh(output)
        return output


class InjectiveGenerator(nn.Module):
    def __init__(self, identity_channels: int = 512):
        super(InjectiveGenerator, self).__init__()
        self.encoder = AttributeEncoder()
        self.generator = DenormalizationGenerator(identity_channels)
    
    def forward(self, input, identity):
        attribute = self.encoder(input)
        output = self.generator(attribute, identity)
        return output, attribute
    
    def attribute(self, input):
        output = self.encoder(input)
        return output
