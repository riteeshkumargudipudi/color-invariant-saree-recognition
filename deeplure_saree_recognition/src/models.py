"""
Deep Metric Learning Architecture with Generalized Mean (GeM) Pooling
and Hyperspherical Bottleneck for Color-Invariant Motif Matching.
Author: Computer Vision Engineering Team
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .transforms import StructuralEdgeExtractor

class GeMPooling(nn.Module):
    """
    Generalized Mean (GeM) Pooling Layer with learnable power parameter p.
    Reference: Radenovic et al., 'Fine-tuning CNN Image Retrieval with No Human Annotation', TPAMI 2018.
    
    Formula:
        f = ( (1 / |Omega|) * sum_{x in Omega} (x^p) ) ^ (1 / p)
    
    Properties:
        - When p = 1: Equivalent to standard Global Average Pooling (GAP).
        - When p -> inf: Approximates Global Max Pooling (GMP).
        - Learnable p (init=3.0) dynamically focuses on salient motif regions
          while preserving spatial coverage of repeating patterns.
    """
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Clamp minimum value to eps to prevent NaN in power operations
        x_clamped = x.clamp(min=self.eps)
        p = self.p.clamp(min=1.0, max=10.0)
        return F.avg_pool2d(x_clamped.pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"

class ColorInvariantSareeNet(nn.Module):
    """
    Lean, high-throughput network for color-invariant textile design recognition.
    
    Pipeline:
      [RGB Input] 
          │
          ▼ (Optional Structural Edge / Luminance Filter)
      [Backbone (EfficientNet-B0 / ResNet34 / ConvNeXt)]
          │
          ▼
      [GeM Pooling (Learnable p)]
          │
          ▼
      [Bottleneck Projection: Linear -> BatchNorm1d -> Dropout]
          │
          ▼
      [L2 Normalization onto Unit Hypersphere S^{d-1}]
          │
          ▼
      [Metric Embedding e in R^{256}]
    """
    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        embedding_dim: int = 256,
        pretrained: bool = True,
        gem_p: float = 3.0,
        color_invariance_mode: str = "hybrid_invariant",
        dropout_rate: float = 0.2
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.embedding_dim = embedding_dim
        self.color_invariance_mode = color_invariance_mode
        
        # Structural edge pre-filter if explicit edge mode selected
        if color_invariance_mode == "structural_edge":
            self.structural_prefilter = StructuralEdgeExtractor()
        else:
            self.structural_prefilter = None

        # Build feature extraction backbone
        self.backbone, num_features = self._build_backbone(backbone_name, pretrained)
        
        # Salient motif pooling
        self.gem_pool = GeMPooling(p=gem_p)
        
        # Hyperspherical metric projection head
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
            nn.Dropout(p=dropout_rate)
        )
        
        # Weight initialization for projection layer
        nn.init.kaiming_normal_(self.bottleneck[1].weight, mode='fan_out', nonlinearity='linear')

    def _build_backbone(self, name: str, pretrained: bool) -> Tuple[nn.Module, int]:
        weights_enum = None
        if pretrained:
            if "efficientnet_b0" in name:
                weights_enum = models.EfficientNet_B0_Weights.DEFAULT
            elif "resnet34" in name:
                weights_enum = models.ResNet34_Weights.DEFAULT
            elif "resnet50" in name:
                weights_enum = models.ResNet50_Weights.DEFAULT
            elif "convnext_tiny" in name:
                weights_enum = models.ConvNeXt_Tiny_Weights.DEFAULT
            else:
                weights_enum = models.EfficientNet_B0_Weights.DEFAULT
        
        try:
            if "efficientnet_b0" in name:
                base = models.efficientnet_b0(weights=weights_enum)
                num_features = base.classifier[1].in_features
                backbone = base.features
            elif "resnet34" in name:
                base = models.resnet34(weights=weights_enum)
                num_features = base.fc.in_features
                backbone = nn.Sequential(*list(base.children())[:-2])
            elif "resnet50" in name:
                base = models.resnet50(weights=weights_enum)
                num_features = base.fc.in_features
                backbone = nn.Sequential(*list(base.children())[:-2])
            elif "convnext_tiny" in name:
                base = models.convnext_tiny(weights=weights_enum)
                num_features = base.classifier[2].in_features
                backbone = base.features
            else:
                base = models.efficientnet_b0(weights=weights_enum)
                num_features = 1280
                backbone = base.features
        except Exception as e:
            print(f"[*] Offline environment detected ({e}). Initializing backbone architecture cleanly without remote weights.")
            base = models.efficientnet_b0(weights=None)
            num_features = 1280
            backbone = base.features
            
        return backbone, num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing L2-normalized metric embeddings.
        Input: [B, 3, H, W]
        Output: [B, embedding_dim] with ||e_i||_2 = 1.0
        """
        # Apply structural decoupling if enabled
        if self.structural_prefilter is not None:
            x = self.structural_prefilter(x)
            
        # Extract spatial feature maps
        features = self.backbone(x)
        
        # GeM pooling -> [B, C, 1, 1]
        pooled = self.gem_pool(features)
        
        # Linear projection
        proj = self.bottleneck(pooled)
        
        # Hyperspherical L2 normalization
        embeddings = F.normalize(proj, p=2, dim=1)
        return embeddings

    def freeze_backbone(self, freeze: bool = True):
        """Freezes/unfreezes backbone parameters for staged warm-up training."""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
            
    def get_parameter_groups(self, base_lr: float, backbone_lr_ratio: float = 0.2):
        """Differential learning rates for backbone vs metric head."""
        return [
            {"params": self.backbone.parameters(), "lr": base_lr * backbone_lr_ratio},
            {"params": self.gem_pool.parameters(), "lr": base_lr},
            {"params": self.bottleneck.parameters(), "lr": base_lr}
        ]
