"""
Color-Invariant Image Transformations and Multi-Scale Structural Extractors.
Author: Computer Vision Engineering Team
"""

import math
import random
from typing import Tuple, Union
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class RandomChannelPermutation:
    """Randomly permutes RGB color channels to destroy chromatic correlation."""
    def __init__(self, p: float = 0.3):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p and img.mode == "RGB":
            channels = list(img.split())
            random.shuffle(channels)
            return Image.merge("RGB", channels)
        return img

class StructuralEdgeExtractor(nn.Module):
    """
    Differentiable PyTorch module that converts RGB batches into 3-channel
    color-invariant structural maps:
      Channel 0: Normalized Luminance (L*)
      Channel 1: Normalized Sobel Gradient Magnitude (motif contours & zari borders)
      Channel 2: High-Pass Spatial Residual (micro-weaves & thread frequency)
    """
    def __init__(self):
        super().__init__()
        # Sobel kernels
        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.],
                                [ 0.,  0.,  0.],
                                [ 1.,  2.,  1.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        
        # Gaussian smoothing kernel for unsharp high-pass
        gauss = torch.tensor([[1., 2., 1.],
                              [2., 4., 2.],
                              [1., 2., 1.]], dtype=torch.float32).view(1, 1, 3, 3) / 16.0
        self.register_buffer("gauss", gauss)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: x in shape [B, 3, H, W] in [0, 1] range.
        Output: struct_x in shape [B, 3, H, W] in [0, 1] range.
        """
        # 1. Luminance Y via standard Rec.601 coefficients
        y = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]  # [B, 1, H, W]
        
        # 2. Spatial Gradient Magnitude (Sobel)
        gx = F.conv2d(y, self.sobel_x, padding=1)
        gy = F.conv2d(y, self.sobel_y, padding=1)
        grad = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        # Normalize gradient per-sample to [0, 1]
        grad_max = torch.amax(grad, dim=(2, 3), keepdim=True) + 1e-6
        grad = grad / grad_max
        
        # 3. High-Pass Spatial Residual (captures fine stitch weave)
        low_pass = F.conv2d(y, self.gauss, padding=1)
        high_pass = torch.abs(y - low_pass)
        hp_max = torch.amax(high_pass, dim=(2, 3), keepdim=True) + 1e-6
        high_pass = high_pass / hp_max
        
        # Stack into 3-channel structural tensor [Y, Gradient, HighPass]
        struct_map = torch.cat([y, grad, high_pass], dim=1)
        return struct_map

def get_training_transforms(img_size: int = 224, color_invariance_mode: str = "hybrid_invariant"):
    """
    Returns training transforms tailored for color-invariant feature learning.
    Applies aggressive chromatic perturbations:
      - Random color jitter (brightness, contrast, saturation, hue across full circle)
      - Random channel permutations
      - Random grayscale fallback
      - Random affine and perspective shifts (folds & draping)
    """
    base_geometric = [
        T.Resize((int(img_size * 1.15), int(img_size * 1.15))),
        T.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.RandomRotation(degrees=15),
    ]
    
    chromatic_augs = [
        RandomChannelPermutation(p=0.35),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.9, hue=0.5),
        T.RandomGrayscale(p=0.30),
        T.RandomAutocontrast(p=0.25),
    ]
    
    normalization = [
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.2, scale=(0.02, 0.2), value="random")
    ]
    
    return T.Compose(base_geometric + chromatic_augs + normalization)

def get_inference_transforms(img_size: int = 224):
    """Deterministic validation/test preprocessing."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

def generate_synthetic_colorway(img: Image.Image) -> Image.Image:
    """
    Simulates a new colorway of an existing saree design by altering
    hue in HSV color space and shifting luminance curves while preserving
    structural motif boundaries.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    hsv = img.convert("HSV")
    h, s, v = hsv.split()
    
    # Shift hue randomly by 60 - 300 degrees
    shift = random.randint(60, 300)
    h_arr = np.array(h, dtype=np.uint8)
    h_arr = ((h_arr.astype(int) + shift) % 256).astype(np.uint8)
    new_h = Image.fromarray(h_arr, mode="L")
    
    # Adjust saturation and contrast
    s_arr = np.array(s, dtype=np.float32)
    s_scale = random.uniform(0.7, 1.3)
    s_arr = np.clip(s_arr * s_scale, 0, 255).astype(np.uint8)
    new_s = Image.fromarray(s_arr, mode="L")
    
    recombined = Image.merge("HSV", (new_h, new_s, v)).convert("RGB")
    return recombined
