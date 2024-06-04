import torch
import torch.nn as nn
import torch.nn.functional as F


class Node2Vec(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int):
        """The Node2Vec model for learning node embeddings.
        It is essentially a simple layer of `nn.Embedding`.

        Args:
            in_dim (int): Input dimension, should be number of nodes.
            emb_dim (int): Embedding dimension.
        """
        super().__init__()
        self.node_embeddings = nn.Embedding(in_dim, emb_dim)
        self._normalize_parameters()

    def _normalize_parameters(self):
        """Normalizes the node embeddings for better convergence."""
        with torch.no_grad():
            self.node_embeddings.weight.data = F.normalize(
                torch.randn_like(self.node_embeddings.weight), dim=1
            )
            self.node_embeddings.weight.requires_grad = True

    def forward(self, x) -> torch.Tensor:
        x = self.node_embeddings(x)
        return x


class SigmoidPredictionHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(torch.sum(u * v, dim=1))
