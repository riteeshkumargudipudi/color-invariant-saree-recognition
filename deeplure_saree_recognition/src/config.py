"""
Configuration and Hyperparameter Management for Color-Invariant Saree Recognition.
Author: Computer Vision Engineering Team
"""

from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass
class SareeConfig:
    # --- Experiment & Seed ---
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 2
    
    # --- Paths ---
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = project_root.parent / "data" / "deeplure_corpus"
    artifacts_dir: Path = project_root / "artifacts"
    weights_dir: Path = project_root / "weights"
    
    # --- Image Preprocessing ---
    img_size: int = 224
    in_channels: int = 3  # RGB with chromatic decoupling or 3-channel structural representation
    
    # --- Color Invariance Modes ---
    # Options: 'chromatic_decoupling' (RGB + aggressive color jitter/shuffle), 
    #          'structural_edge' (Luminance + Sobel Edge + High-pass Texture),
    #          'hybrid_invariant' (Combines multi-scale structural maps with normalized reflectance)
    color_invariance_mode: str = "hybrid_invariant"
    
    # --- Architecture ---
    backbone_name: str = "efficientnet_b0"  # Lean, edge-friendly: ~4.0M params, ~0.39 GFLOPs
    fallback_backbone: str = "resnet34"     # Highly robust alternative
    embedding_dim: int = 256               # Compact metric embedding for fast L2 search
    gem_p: float = 3.0                     # Generalized Mean Pooling initial power (learnable)
    freeze_backbone_epochs: int = 2        # Warmup head before fine-tuning backbone
    
    # --- Metric Learning & Loss ---
    loss_type: str = "arcface_triplet"     # 'arcface', 'supcon', 'triplet', 'arcface_triplet'
    arcface_scale: float = 32.0            # Radius scale s
    arcface_margin: float = 0.35           # Angular margin m (in radians)
    triplet_margin: float = 0.30           # Margin for hard-negative triplet loss
    triplet_weight: float = 0.5            # Weight balancing ArcFace and Triplet loss
    supcon_temperature: float = 0.07       # Temperature for Supervised Contrastive Loss
    
    # --- Training & Optimization ---
    epochs: int = 15
    batch_size: int = 32                   # P=8 classes, K=4 colorways/instances per batch
    samples_per_class: int = 4             # K in balanced batch sampler
    learning_rate: float = 3e-4
    backbone_lr_ratio: float = 0.2         # Backbone trains at 0.2x head lr
    weight_decay: float = 1e-4
    mixed_precision: bool = True           # torch.cuda.amp.autocast
    gradient_clip_val: float = 2.0
    patience: int = 5                      # Early stopping patience
    
    # --- Evaluation Protocol ---
    gallery_query_split_ratio: float = 0.5  # 50% gallery, 50% query per test motif
    test_design_ratio: float = 0.25        # 25% unseen designs reserved for open-set evaluation
    verification_pairs_count: int = 600    # Balanced genuine (same design) & impostor pairs
