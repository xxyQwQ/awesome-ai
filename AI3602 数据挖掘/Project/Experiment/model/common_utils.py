from ogb.nodeproppred import Evaluator
import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_curve
from sklearn.metrics import auc



class Node2Vec(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, path):
        """Save the model parameters to the specified path."""
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path, device, *args, **kwargs):
        """Load the model parameters from the specified path."""
        self.load_state_dict(torch.load(path, map_location=device))
        print(f"Model loaded from {path}")
        return self

    @torch.no_grad()
    def embed_all(self, data):
        emb_list = []
        for ids in tqdm(range(0, data['graph'].num_nodes)):
            emb = self.forward([ids]).cpu()
            emb_list.append(emb)
        emb = torch.cat(emb_list, dim=0)
        return emb


class Classifier(nn.Module):
    def __init__(self, in_dim, num_cls, hidden_dim=256):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_cls)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def evaluate(pred, label):
    input_dict = {"y_true": label, "y_pred": pred}
    return Evaluator(name='ogbn-arxiv').eval(input_dict)


def calc_auc_score(predictions, truth):
    fpr, tpr, _ = roc_curve(truth, predictions)
    return auc(fpr, tpr)


class NegativeSamplingLoss(nn.Module):
    """The negative sampling loss function.

    Args:
        eps (float, optional): For numerical stability. Defaults to 1e-9.
    """

    def __init__(self, eps: float = 1e-9):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        cur_embs: torch.Tensor,
        pos_embs: torch.Tensor,
        neg_embs: torch.Tensor,
    ):
        """
        Compute the negative sampling loss.

        Args:
            cur_embs (torch.Tensor): Embeddings of the current nodes, shape (B, H).
            pos_embs (torch.Tensor): Embeddings of the positive samples, shape (B, N_pos, H).
            neg_embs (torch.Tensor): Embeddings of the negative samples, shape (B, N_neg, H).

        Returns:
            torch.Tensor: The negative sampling loss.
        """
        cur_embs = F.normalize(cur_embs, p=2, dim=1)  # shape (B, H)
        pos_embs = F.normalize(pos_embs, p=2, dim=2)  # shape (B, N_pos, H)
        neg_embs = F.normalize(neg_embs, p=2, dim=2)  # shape (B, N_neg, H)

        # Reshape embeddings for broadcasting
        cur_embs = cur_embs.unsqueeze(1)  # shape (B, 1, H)
        
        # Compute scores. Actually cos_sim
        pos_scores = torch.bmm(pos_embs, cur_embs.transpose(1, 2)).squeeze(2)  # shape (B, N_pos)
        neg_scores = torch.bmm(neg_embs, cur_embs.transpose(1, 2)).squeeze(2)  # shape (B, N_neg)

        # Compute positive and negative losses
        pos_loss = (1-pos_scores).mean()
        neg_loss = (neg_scores).mean()

        # Total loss
        loss = pos_loss + neg_loss

        return loss



# Example usage
if __name__ == "__main__":
    loss_fn = NegativeSamplingLoss()
    cur_embs = torch.randn(8, 128)  # Example current node embeddings
    pos_embs = torch.randn(8, 5, 128)  # Example positive samples
    neg_embs = torch.randn(8, 20, 128)  # Example negative samples

    loss = loss_fn(cur_embs, pos_embs, neg_embs)
    print("Loss:", loss.item())