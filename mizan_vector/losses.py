"""
Loss functions for training Mizan-based embedding models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def mizan_similarity_torch(
    v1: torch.Tensor,
    v2: torch.Tensor,
    p: float = 2.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Batched Mizan similarity for vectors.
    v1, v2: [batch, dim]
    Returns: [batch] similarity in [0, 1]
    """
    diff = torch.norm(v1 - v2, p=2, dim=-1)
    norm1 = torch.norm(v1, p=2, dim=-1)
    norm2 = torch.norm(v2, p=2, dim=-1)

    num = diff ** p
    den = norm1 ** p + norm2 ** p + eps

    return 1.0 - num / den


class MizanContrastiveLoss(nn.Module):
    """
    Contrastive loss using Mizan similarity.

    Positive pairs → high similarity
    Negative pairs → similarity < margin
    """

    def __init__(self, margin: float = 0.5, p: float = 2.0, eps: float = 1e-8):
        super().__init__()
        self.margin = margin
        self.p = p
        self.eps = eps

    def forward(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
        labels: torch.Tensor,   # 1 = positive, 0 = negative
    ) -> torch.Tensor:
        sim = mizan_similarity_torch(emb1, emb2, p=self.p, eps=self.eps)

        labels = labels.float()

        pos_loss = labels * (1.0 - sim)
        neg_loss = (1.0 - labels) * F.relu(sim - self.margin)

        return (pos_loss + neg_loss).mean()


class MizanTripletLoss(nn.Module):
    """
    Triplet loss using Mizan similarity:
    anchor-positive similarity > anchor-negative similarity
    """

    def __init__(self, margin: float = 0.3, p: float = 2.0):
        super().__init__()
        self.margin = margin
        self.p = p

    def forward(self, anchor, positive, negative):
        sim_pos = mizan_similarity_torch(anchor, positive, p=self.p)
        sim_neg = mizan_similarity_torch(anchor, negative, p=self.p)

        loss = F.relu(self.margin - (sim_pos - sim_neg))
        return loss.mean()
