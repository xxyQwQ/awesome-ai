import torch
from torch import nn
from model.encoder import StructureEncoder, StyleEncoder


class DenormalizationLayer(nn.Module):
    def __init__(self, input_channels: int, structure_channels: int, style_channels: int):
        super(DenormalizationLayer, self).__init__()
        self.norm = nn.BatchNorm2d(input_channels)

        self.structure_mu = nn.Conv2d(structure_channels, input_channels, 1, 1, 0)
        self.structure_sigma = nn.Conv2d(structure_channels, input_channels, 1, 1, 0)

        self.style_mu = nn.Linear(style_channels, input_channels)
        self.style_sigma = nn.Linear(style_channels, input_channels)

        self.input_mask = nn.Conv2d(input_channels, 1, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input, structure, style):
        input = self.norm(input)

        structure_mu = self.structure_mu(structure)
        structure_sigma = self.structure_sigma(structure)
        structure_activation = structure_mu + structure_sigma * input

        target_shape = (input.shape[0], input.shape[1], 1, 1)
        style_mu = self.style_mu(style).reshape(target_shape).expand_as(input)
        style_sigma = self.style_sigma(style).reshape(target_shape).expand_as(input)
        style_activation = style_mu + style_sigma * input

        input_mask = self.sigmoid(self.input_mask(input))
        output = (1 - input_mask) * structure_activation + input_mask * style_activation
        return output


class DenormalizationBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, structure_channels: int, style_channels: int):
        super(DenormalizationBlock, self).__init__()
        self.transform = bool(input_channels != output_channels)
        self.relu = nn.ReLU(inplace=True)

        self.denorm_1 = DenormalizationLayer(input_channels, structure_channels, style_channels)
        self.conv_1 = nn.Conv2d(input_channels, input_channels, 3, 1, 1)

        self.denorm_2 = DenormalizationLayer(input_channels, structure_channels, style_channels)
        self.conv_2 = nn.Conv2d(input_channels, output_channels, 3, 1, 1)

        if self.transform:
            self.denorm_3 = DenormalizationLayer(input_channels, structure_channels, style_channels)
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
    def __init__(self):
        super(DenormalizationGenerator, self).__init__()
        self.num_blocks = 7

        self.boot = nn.ConvTranspose2d(512, 512, 2, 1, 0)
        self.tanh = nn.Tanh()

        result_channels = [512, 512, 512, 512, 256, 128, 64, 1]
        structure_channels = [512, 1024, 512, 256, 128, 64, 64]
        style_channels = [512, 512, 512, 512, 512, 512, 512]

        denorm_list = []
        for i in range(self.num_blocks):
            denorm_list.append(DenormalizationBlock(result_channels[i], result_channels[i + 1], structure_channels[i], style_channels[i]))
        self.denorm = nn.ModuleList(denorm_list)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, structure, style):
        target_shape = (style.shape[0], style.shape[1], 1, 1)
        result = self.boot(style.reshape(target_shape))

        for i in range(self.num_blocks):
            result = self.denorm[i](result, structure[i], style)
            if i < self.num_blocks - 1:
                result = self.upsample(result)

        result = self.tanh(result)
        return result


class SynthesisGenerator(nn.Module):
    def __init__(self, reference_count: int = 1):
        super(SynthesisGenerator, self).__init__()
        self.structure_encoder = StructureEncoder()
        self.style_encoder = StyleEncoder(reference_count)
        self.result_decoder = DenormalizationGenerator()

    def forward(self, reference, template):
        structure = self.structure_encoder(template)
        style = self.style_encoder(reference)
        result = self.result_decoder(structure, style)
        return result, structure, style

    def structure(self, input):
        output = self.structure_encoder(input)
        return output

    def style(self, input):
        output = self.style_encoder(input)
        return output
