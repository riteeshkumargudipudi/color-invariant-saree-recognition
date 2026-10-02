"""
Metric Learning Loss Functions for Color-Invariant Feature Space Optimization.
Implements:
  1. ArcFace (Additive Angular Margin Loss)
  2. Supervised Contrastive Loss (SupCon)
  3. Batch-Hard Triplet Loss with Color-Adversarial Mining
Author: Computer Vision Engineering Team
"""

import math
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcFaceLoss(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss for Deep Textile/Motif Verification.
    Reference: Deng et al., CVPR 2019.
    
    Formula:
      L = -log( e^{s * cos(theta_{y_i} + m)} / ( e^{s * cos(theta_{y_i} + m)} + sum_{j != y_i} e^{s * cos(theta_j)} ) )
      
    Args:
      num_classes: Number of distinct saree design categories.
      embedding_dim: Dimension of metric embedding (e.g. 256).
      scale (s): Radius of hypersphere (typically 30.0 - 64.0).
      margin (m): Angular margin in radians (typically 0.2 - 0.5 rad ~ 11.5 - 28.6 deg).
    """
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 256,
        scale: float = 32.0,
        margin: float = 0.35,
        easy_margin: bool = False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin

        # Normalized class prototype weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        # Precompute constants for angular addition: cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
          embeddings: [B, D] L2-normalized embeddings (||x|| = 1)
          labels: [B] Class integer IDs
        Returns:
          Cross-entropy loss with angular additive margin.
        """
        # Normalize class prototype weights
        w_norm = F.normalize(self.weight, p=2, dim=1)
        # Cosine similarity between embeddings and class prototypes: [B, num_classes]
        cosine = F.linear(embeddings, w_norm).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        
        # sin(theta) = sqrt(1 - cos^2(theta))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)).clamp(0, 1.0)
        
        # cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot mask for ground-truth class
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        
        # Apply margin only to target class logit
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        
        return F.cross_entropy(output, labels)


class BatchHardTripletLoss(nn.Module):
    """
    Batch-Hard Triplet Margin Loss with Cosine / Euclidean Distance.
    Reference: Hermans et al., 'In Defense of the Triplet Loss for Person Re-ID', 2017.
    
    For each anchor:
      - Mine hardest positive (same design in different colorway, max distance).
      - Mine hardest negative (different design, min distance / similar background palette!).
    """
    def __init__(self, margin: float = 0.3, distance_metric: str = "cosine"):
        super().__init__()
        self.margin = margin
        self.distance_metric = distance_metric

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        embeddings: [B, D] L2-normalized embeddings
        labels: [B] integer class labels
        """
        # Pairwise distance matrix
        if self.distance_metric == "cosine":
            # Cosine distance = 1 - sim in [0, 2]
            sim = torch.matmul(embeddings, embeddings.T)
            dist = 1.0 - sim
        else:
            # Squared Euclidean distance on unit sphere = 2 * (1 - sim)
            dist = torch.cdist(embeddings, embeddings, p=2)

        # Mask matrices: positive mask (same class) and negative mask (diff class)
        labels = labels.unsqueeze(1)
        is_pos = torch.eq(labels, labels.T)
        is_neg = torch.ne(labels, labels.T)

        # Hardest positive: maximum distance among positives (excluding self)
        dist_pos = dist * is_pos.float()
        # Set diagonal to zero
        dist_pos = dist_pos - torch.diag(torch.diag(dist_pos))
        hardest_pos, _ = torch.max(dist_pos, dim=1)

        # Hardest negative: minimum distance among negatives
        # Add high value to positive pairs so they aren't chosen as min
        dist_neg = dist + 1e5 * is_pos.float()
        hardest_neg, _ = torch.min(dist_neg, dim=1)

        # Triplet margin loss: max(0, d(a, p) - d(a, n) + margin)
        loss = F.relu(hardest_pos - hardest_neg + self.margin)
        return loss.mean()


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Learning (SupCon) Loss.
    Reference: Khosla et al., NeurIPS 2020.
    
    Pulls all samples of the same design (regardless of colorway) together on the unit hypersphere,
    while pushing different designs apart.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = embeddings.device
        batch_size = embeddings.shape[0]
        
        # Cosine similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # For numerical stability, subtract max
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_max.detach()
        
        # Create masks
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Mask out self-contrast
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Exponentiate
        exp_sim = torch.exp(sim_matrix) * logits_mask
        log_prob = sim_matrix - torch.log(exp_sim.sum(1, keepdim=True) + 1e-7)
        
        # Mean log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-7)
        loss = -mean_log_prob_pos.mean()
        return loss
