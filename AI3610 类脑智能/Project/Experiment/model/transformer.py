from torch import nn
from einops.layers.torch import Reduce
from spikingjelly.activation_based import layer, neuron

from model.embedding import PatchEmbedding
from model.attention import FeedForward, SelfAttention


class EncoderBlock(nn.Module):
    def __init__(self, num_heads=8, num_ceils=8, num_channels=512):
        super().__init__()
        self.self_attention = SelfAttention(num_heads, num_ceils, num_channels)
        self.feed_forward = FeedForward(num_channels, num_channels)

    def forward(self, input):
        output = self.self_attention(input)
        output = self.feed_forward(output)
        return output


class SpikingTransformer(nn.Module):
    def __init__(self, in_channels=2, image_size=128, num_classes=10, num_layers=8, num_heads=8, num_channels=512):
        super().__init__()
        num_poolings = 2 if image_size <= 32 else 3 if image_size <= 128 else 4
        num_ceils = image_size // (2 ** num_poolings)
        self.patch_embedding = PatchEmbedding(in_channels, num_channels, num_poolings)
        self.encoder_block = nn.Sequential(
            *[EncoderBlock(num_heads, num_ceils, num_channels) for _ in range(num_layers)]
        )
        self.classifier_head = nn.Sequential(
            Reduce('T B C H W -> T B C', 'mean'),
            neuron.LIFNode(detach_reset=True),
            layer.Linear(num_channels, num_classes),
            Reduce('T B C -> B C', 'mean')
        )

    def forward(self, input):
        output = self.patch_embedding(input)
        output = self.encoder_block(output)
        output = self.classifier_head(output)
        return output
