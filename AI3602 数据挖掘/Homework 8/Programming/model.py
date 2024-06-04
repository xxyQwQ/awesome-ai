import torch
import torch.nn as nn

from configs import Config
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Optional


def get_activation(activation: str):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"Activation function {activation} is not supported.")


class BiLSTMEncoder(nn.Module):
    """
    A bidirectional LSTM encoder for extracting features from sequences.

    Sequence features are pooled using maxpooling.
    """

    def __init__(
        self,
        n_tokens: int = 10000,
        embedding_dim: int = 100,
        hidden_dim: int = 200,
        num_layers: int = 1,
        dropout: Optional[float] = None,
    ):
        super().__init__()

        self.embedding = nn.Embedding(n_tokens, embedding_dim)
        self.rnn_unit = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
        )
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor,
    ):
        x = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]

        lengths = padding_mask.sum(dim=1).long().cpu()
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # output: packed sequence
        # h: [2 * num_layers, B, hidden_dim]
        # c: [2 * num_layers, B, hidden_dim]
        output, (h, c) = self.rnn_unit(x)
        # output: [batch_size, seq_len, 2 * hidden_dim]
        output, _ = pad_packed_sequence(output, batch_first=True)

        # apply padding mask
        output = output * padding_mask.unsqueeze(-1)

        # maxpool, output: [batch_size, 2 * hidden_dim]
        output, _ = torch.max(output, dim=1)

        # dropout
        if self.dropout is not None:
            output = self.dropout(output)

        return output


class EmbeddingEncoder(nn.Module):
    """
    A simple embedding encoder for extracting features from sequences.

    Sequence features are pooled using maxpooling.

    NOTE: This encoder is the "MLP" used for extracting token features in CodeNN.
    While it is referred to as an "MLP", the original CodeNN implementation only
    uses a single linear layer with an activation function.
    """

    def __init__(
        self,
        n_tokens: int = 10000,
        embedding_dim: int = 100,
        activation: str = "tanh",
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(n_tokens, embedding_dim)
        self.activation = get_activation(activation)

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor,
    ):
        x = self.embedding(input_ids)  # B, seq_len, embedding_dim

        # apply padding mask
        x = x * padding_mask.unsqueeze(-1)

        # maxpool, output: [batch_size, embedding_dim]
        x, _ = torch.max(x, dim=1)

        # activation
        x = self.activation(x)

        # dropout
        if self.dropout is not None:
            x = self.dropout(x)

        return x


class CodeNN(nn.Module):
    """
    The CodeNN model for code search tasks.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # sanity check
        lstm_out_dim = 2 * config.lstm_n_layers * config.lstm_hidden_dim
        if config.fusion_hidden_dim != lstm_out_dim:
            raise ValueError(
                "fusion_hidden_dim must be equal to 2 * lstm_n_layers * lstm_hidden_dim"
                f" ({lstm_out_dim}), but got {config.fusion_hidden_dim}."
            )

        # TODO (Task 2.1)
        # Complete the model architecture.
        # You need to implement the following components:
        # - method_name_encoder
        # - api_seq_encoder
        # - token_encoder
        #
        # NOTE: the fusion_mlp and desc_encoder are already implemented.
        #
        # Hints:
        # 1. We have provided the BiLSTMEncoder and EmbeddingEncoder classes.
        # 2. Use the hyperparams from the config to initialize the components.
        # 3. Use `get_activation` to get the activation function.

        self.method_name_encoder = BiLSTMEncoder(
            n_tokens=config.n_tokens,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.lstm_hidden_dim,
            num_layers=config.lstm_n_layers,
            dropout=config.dropout,
        )
        self.api_seq_encoder = BiLSTMEncoder(
            n_tokens=config.n_tokens,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.lstm_hidden_dim,
            num_layers=config.lstm_n_layers,
            dropout=config.dropout,
        )
        self.token_encoder = EmbeddingEncoder(
            n_tokens=config.n_tokens,
            embedding_dim=config.embedding_dim,
            activation=config.token_activation,
            dropout=config.dropout,
        )

        # NOTE: We have defined the fusion mlp and the description encoder for you.
        # But you can still modify them if you want to.
        # code feature fusion
        self.fusion_mlp = nn.Sequential(
            nn.Linear(
                # 2 * (bidirectional) * 2 (lstm encoders) + 1 (token encoder)
                in_features=2 * 2 * config.lstm_hidden_dim + config.embedding_dim,
                out_features=config.fusion_hidden_dim,
            ),
            get_activation(config.fusion_activation),
            nn.Dropout(config.dropout) if config.dropout is not None else nn.Identity(),
        )

        # natural language features
        self.desc_encoder = BiLSTMEncoder(
            n_tokens=config.n_tokens,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.lstm_hidden_dim,
            num_layers=config.lstm_n_layers,
            dropout=config.dropout,
        )

        # End of TODO

    def code_features(
        self,
        method_ids: torch.Tensor,
        method_padding_mask: torch.Tensor,
        token_ids: torch.Tensor,
        token_padding_mask: torch.Tensor,
        apiseq_ids: torch.Tensor,
        apiseq_padding_mask: torch.Tensor,
    ):
        # TODO (Task 2.1)
        # Complete the forward pass for extracting code features.
        # 1. Extract features from the method name, tokens, and API sequences.
        # 2. Concatenate the features and pass them through the fusion MLP.
        # 3. Return the final code features.

        method_features = self.method_name_encoder(method_ids, method_padding_mask)
        token_features = self.token_encoder(token_ids, token_padding_mask)
        apiseq_features = self.api_seq_encoder(apiseq_ids, apiseq_padding_mask)
        code_features = self.fusion_mlp(
            torch.cat([method_features, token_features, apiseq_features], dim=-1)
        )

        # End of TODO

        return code_features

    def desc_features(self, desc_ids: torch.Tensor, desc_padding_mask: torch.Tensor):
        # TODO (Task 2.1)
        # Implement the forward pass for extracting natural language features.

        desc_features = self.desc_encoder(desc_ids, desc_padding_mask)

        # End of TODO

        return desc_features

    def forward(
        self,
        method_ids: torch.Tensor,
        method_padding_mask: torch.Tensor,
        token_ids: torch.Tensor,
        token_padding_mask: torch.Tensor,
        apiseq_ids: torch.Tensor,
        apiseq_padding_mask: torch.Tensor,
        pos_desc_ids: torch.Tensor,
        pos_desc_padding_mask: torch.Tensor,
        neg_desc_ids: torch.Tensor,
        neg_desc_padding_mask: torch.Tensor,
    ):
        code_features = self.code_features(
            method_ids,
            method_padding_mask,
            token_ids,
            token_padding_mask,
            apiseq_ids,
            apiseq_padding_mask,
        )
        pos_desc_features = self.desc_features(pos_desc_ids, pos_desc_padding_mask)
        neg_desc_features = self.desc_features(neg_desc_ids, neg_desc_padding_mask)

        return code_features, pos_desc_features, neg_desc_features
