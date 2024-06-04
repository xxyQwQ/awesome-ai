from model.common_utils import Node2Vec
import torch.nn as nn
import torch
import hashlib



def encode_node_id(node_id):
    node_id_str = str(node_id)
    
    hash_functions = [
        hashlib.md5,
        hashlib.sha1,
        hashlib.sha256,
        hashlib.sha512
        ]
    
    hash_values = []
    
    for hash_func in hash_functions:
        hash_object = hash_func(node_id_str.encode())
        hash_bytes = hash_object.digest()
        hash_ints = [b for b in hash_bytes]
        hash_values.extend(hash_ints)
    
    vector = torch.tensor(hash_values, dtype=torch.float32) / 255.0
    
    return vector


class MLP(Node2Vec):
    
    def __init__(self, in_dim=132, embedding_dim=768, hidden=4096, device='cuda:0'):
        super().__init__()
        self.device = device
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LeakyReLU(),
            # nn.Linear(hidden, hidden),
            # nn.LeakyReLU(),
            nn.Linear(hidden, embedding_dim)
        ).to(device)

    def forward(self, x):
        x = [encode_node_id(node_id) for node_id in (x.cpu().numpy() if type(x) == torch.Tensor else x)]
        x = torch.stack(x).to(self.device)
        x = self.mlp(x)
        return x

