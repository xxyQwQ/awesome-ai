import numpy as np
from einops import rearrange
import torch
from torch import nn, einsum
from torch.nn import functional as F


class SelfAttention(nn.Module):
    def __init__(self, embedding_dims=256, hidden_dims=512, num_heads=8, dropout=0.2, max_length=1024):
        super(SelfAttention, self).__init__()
        self.hidden_dims = hidden_dims
        self.num_heads = num_heads
        self.projection = nn.Linear(embedding_dims, 3 * hidden_dims)
        self.transform = nn.Linear(hidden_dims, embedding_dims)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", nn.Transformer.generate_square_subsequent_mask(max_length).view(1, 1, max_length, max_length))

    def forward(self, input):
        sequence_length = input.shape[0]
        query, key, value = rearrange(self.projection(input), 'n b (k h d) -> k b h n d', k=3, h=self.num_heads)
        attention = einsum('b h i d, b h j d -> b h i j', query, key) / np.sqrt(self.hidden_dims)
        attention += self.mask[:, :, :sequence_length, :sequence_length]
        attention = self.dropout(F.softmax(attention, dim=-1))
        output = einsum('b h i d, b h d j -> b h i j', attention, value)
        output = rearrange(output, 'b h n d -> n b (h d)')
        return self.dropout(self.transform(output))


class FeedForward(nn.Module):
    def __init__(self, embedding_dims=256, hidden_dims=512, dropout=0.2):
        super(FeedForward, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dims, hidden_dims),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, embedding_dims),
            nn.Dropout(dropout)
        )

    def forward(self, input):
        output = self.network(input)
        return output


class DecoderBlock(nn.Module):
    def __init__(self, embedding_dims=256, hidden_dims=512, num_heads=8, dropout=0.2, max_length=1024):
        super(DecoderBlock, self).__init__()
        self.former_norm = nn.LayerNorm(embedding_dims)
        self.latter_norm = nn.LayerNorm(embedding_dims)
        self.self_attention = SelfAttention(embedding_dims, hidden_dims, num_heads, dropout, max_length)
        self.feed_forward = FeedForward(embedding_dims, hidden_dims, dropout)

    def forward(self, input):
        output = self.self_attention(self.former_norm(input)) + input
        output = self.feed_forward(self.latter_norm(output)) + output
        return output


class GPT(nn.Module):
    def __init__(self, num_tokens, embedding_dims=256, hidden_dims=512, num_heads=8, num_layers=2, dropout=0.2, max_length=1024):
        super(GPT, self).__init__()
        self.norm = nn.LayerNorm(embedding_dims)
        self.dropout = nn.Dropout(dropout)
        self.token_embedding = nn.Embedding(num_tokens, embedding_dims)
        self.position_embedding = nn.Parameter(torch.rand(max_length, 1, embedding_dims))
        self.block_list = nn.ModuleList([DecoderBlock(embedding_dims, hidden_dims, num_heads, dropout, max_length) for _ in range(num_layers)])
        self.head = nn.Linear(embedding_dims, num_tokens)
        self.init_weight()

    def forward(self, input):
        sequence_length = input.shape[0]
        embedding = self.token_embedding(input) + self.position_embedding[:sequence_length]
        output = self.dropout(embedding)
        for decoder_block in self.block_list:
            output = decoder_block(output)
        output = self.head(self.norm(output))
        return F.log_softmax(output, dim=-1)

    def init_weight(self):
        nn.init.uniform_(self.token_embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.head.weight, -0.1, 0.1)
        nn.init.zeros_(self.head.bias)
