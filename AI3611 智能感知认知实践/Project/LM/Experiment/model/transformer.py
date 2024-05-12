import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_dims=256, dropout=0.2, max_length=1024):
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        embedding = torch.zeros(max_length, embedding_dims)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, embedding_dims, 2, dtype=torch.float) * (-np.log(10000.0) / embedding_dims))
        embedding[:, 0::2] = torch.sin(position * divisor)
        embedding[:, 1::2] = torch.cos(position * divisor)
        embedding = embedding.unsqueeze(0).transpose(0, 1)
        self.register_buffer('embedding', embedding)

    def forward(self, input):
        sequence_length = input.shape[0]
        output = input + self.embedding[:sequence_length]
        return self.dropout(output)


class Transformer(nn.Module):
    def __init__(self, num_tokens, embedding_dims=256, hidden_dims=512, num_heads=8, num_layers=2, dropout=0.2, max_length=1024):
        super(Transformer, self).__init__()
        self.emdedding_dims = embedding_dims
        self.input_mask = None
        self.position_encoder = PositionalEmbedding(embedding_dims, dropout, max_length)
        encoder_layer = TransformerEncoderLayer(embedding_dims, num_heads, hidden_dims, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        self.encoder = nn.Embedding(num_tokens, embedding_dims)
        self.decoder = nn.Linear(embedding_dims, num_tokens)
        self.init_weight()

    def forward(self, input):
        if self.input_mask is None or self.input_mask.shape[0] != len(input):
            self.input_mask = nn.Transformer.generate_square_subsequent_mask(len(input)).to(input.device)
        embedding = self.encoder(input) * np.sqrt(self.emdedding_dims)
        output = self.position_encoder(embedding)
        output = self.transformer_encoder(output, self.input_mask)
        return F.log_softmax(self.decoder(output), dim=-1)

    def init_weight(self):
        nn.init.uniform_(self.encoder.weight, -0.1, 0.1)
        nn.init.uniform_(self.decoder.weight, -0.1, 0.1)
        nn.init.zeros_(self.decoder.bias)
