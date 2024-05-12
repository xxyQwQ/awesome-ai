from torch import nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, num_tokens, embedding_dims=256, hidden_dims=512, num_layers=2, dropout=0.2, tied=False, cell='rnn'):
        super(RNN, self).__init__()
        self.num_tokens = num_tokens
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.cell = cell
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.Embedding(num_tokens, embedding_dims)
        self.decoder = nn.Linear(hidden_dims, num_tokens)
        self.rnn = getattr(nn, cell.upper())(embedding_dims, hidden_dims, num_layers, dropout=dropout)
        if tied:
            assert embedding_dims == hidden_dims
            self.decoder.weight = self.encoder.weight
        self.init_weight()

    def forward(self, input, hidden):
        embedding = self.dropout(self.encoder(input))
        output, hidden = self.rnn(embedding, hidden)
        output = self.decoder(self.dropout(output))
        return F.log_softmax(output, dim=-1), hidden

    def init_weight(self):
        nn.init.uniform_(self.encoder.weight, -0.1, 0.1)
        nn.init.uniform_(self.decoder.weight, -0.1, 0.1)
        nn.init.zeros_(self.decoder.bias)

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        if self.cell == 'lstm':
            return (weight.new_zeros(self.num_layers, batch_size, self.hidden_dims), weight.new_zeros(self.num_layers, batch_size, self.hidden_dims))
        else:
            return weight.new_zeros(self.num_layers, batch_size, self.hidden_dims)
