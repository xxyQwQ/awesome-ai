import torch
import torch.nn as nn


class SimilarityLoss(nn.Module):
    def __init__(
        self, margin: float = 0.05, eps: float = 1e-8, return_sims: bool = False
    ):
        super().__init__()
        self.margin = margin
        self.eps = eps
        self.return_sims = return_sims

    def forward(
        self,
        code_features: torch.Tensor,
        pos_desc_features: torch.Tensor,
        neg_desc_features: torch.Tensor,
    ):
        # TODO (Task 2.2)
        # Implement the ranking loss function.
        #
        # compute positive and negative cosine similarities
        # and then calculate the loss value

        cos_sim = nn.CosineSimilarity(dim=-1, eps=self.eps)
        pos_sim = cos_sim(code_features, pos_desc_features)
        neg_sim = cos_sim(code_features, neg_desc_features)
        loss = torch.clamp(self.margin - pos_sim + neg_sim, min=0).mean()

        # End of TODO

        if self.return_sims:
            return loss, pos_sim.mean(), neg_sim.mean()

        return loss
