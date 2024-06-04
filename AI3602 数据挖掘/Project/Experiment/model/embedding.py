from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertModel
from model.common_utils import Node2Vec
import torch.nn as nn
import torch.nn.functional as F
import torch


class Embedding(Node2Vec):
    
    def __init__(self, num_node, embedding_dim=768, device='cuda:0'):
        super().__init__()
        self.embedding = nn.Embedding(num_node, embedding_dim, device=device)
        self._normalize_parameters()

    def _normalize_parameters(self):
        """Normalizes the node embeddings for better convergence."""
        with torch.no_grad():
            self.embedding.weight.data = F.normalize(
                torch.randn_like(self.embedding.weight), dim=1
            )
            self.embedding.weight.requires_grad = True

    def forward(self, x):
        x = self.embedding(x)
        return x

