"""
Saree Dataset Management, Multi-Colorway Pairing, and Balanced Batch Sampling.
Author: Computer Vision Engineering Team
"""

import os
import glob
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .transforms import (
    get_training_transforms,
    get_inference_transforms,
    generate_synthetic_colorway
)

class ColorInvariantSareeDataset(Dataset):
    """
    Textile dataset for color-invariant saree design recognition.
    
    Supports:
      1. Real multi-image design clusters.
      2. Dynamic colorway augmentation: simulates unseen fabric dyes and zari palettes
         to enforce strict motif invariance during training.
    """
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform=None,
        is_training: bool = True,
        synthesize_colorways: bool = True
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_training = is_training
        self.synthesize_colorways = synthesize_colorways
        
        # Build mapping from class_id -> list of image indices
        self.class_to_indices = defaultdict(list)
        for idx, lbl in enumerate(labels):
            self.class_to_indices[lbl].append(idx)
            
        self.classes = sorted(list(self.class_to_indices.keys()))
        self.num_classes = len(self.classes)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path = self.image_paths[idx]
        label = self.labels[idx]
        
        with Image.open(path) as img:
            img = img.convert("RGB")
            
            # During training, with probability 0.5, synthesize an alternate colorway
            # to expose the network to diverse palette renderings of the same motif
            if self.is_training and self.synthesize_colorways and random.random() < 0.5:
                img = generate_synthetic_colorway(img)
                
            if self.transform is not None:
                tensor_img = self.transform(img)
            else:
                tensor_img = get_inference_transforms()(img)
                
        return tensor_img, label, os.path.basename(path)


class BalancedBatchSampler(Sampler):
    """
    (P x K) Balanced Batch Sampler for Metric Learning.
    
    In each mini-batch:
      - Randomly samples P unique design classes.
      - Randomly samples K instances/colorways for each chosen class.
      - Batch size = P * K.
    
    Guarantees that every batch contains both multiple positive pairs (same design)
    and hard negative pairs (different designs), essential for ArcFace & Triplet mining.
    """
    def __init__(
        self,
        labels: List[int],
        num_classes_per_batch: int = 8,
        samples_per_class: int = 4
    ):
        self.labels = labels
        self.num_classes_per_batch = num_classes_per_batch
        self.samples_per_class = samples_per_class
        self.batch_size = num_classes_per_batch * samples_per_class

        self.class_to_indices = defaultdict(list)
        for idx, lbl in enumerate(labels):
            self.class_to_indices[lbl].append(idx)
            
        self.classes = list(self.class_to_indices.keys())
        self.num_batches = len(labels) // self.batch_size

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            selected_classes = random.sample(self.classes, min(self.num_classes_per_batch, len(self.classes)))
            for cls in selected_classes:
                indices = self.class_to_indices[cls]
                if len(indices) >= self.samples_per_class:
                    selected_indices = random.sample(indices, self.samples_per_class)
                else:
                    # Oversample with replacement if fewer than K samples exist
                    selected_indices = random.choices(indices, k=self.samples_per_class)
                batch.extend(selected_indices)
            yield batch

    def __len__(self) -> int:
        return max(1, self.num_batches)


def load_saree_corpus(data_dir: str) -> Tuple[List[str], List[int], Dict[str, str]]:
    """
    Discovers all saree images from the dataset directory and organizes them into
    design clusters based on directory structure and filename prefixes.
    """
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    all_files = []
    for ext in extensions:
        all_files.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
        
    all_files = sorted(list(set(all_files)))
    if not all_files:
        raise FileNotFoundError(f"No saree images found in directory: {data_dir}")
        
    # Group images into design clusters:
    # 1. Subfolder names (e.g. handloom_sarees, normal_sarees, ikat, banarasi)
    # 2. Design hash/prefix patterns
    design_to_paths = defaultdict(list)
    for p in all_files:
        parent = os.path.basename(os.path.dirname(p))
        base = os.path.basename(p)
        
        # Check if filename contains design family tag (e.g. h_img vs img, or numeric range)
        # We group into fine-grained design clusters
        prefix = base.split("_")[0] if "_" in base else "design"
        num_part = "".join([c for c in base if c.isdigit()])
        # Bin numeric IDs into design families if available, else group by category
        if len(num_part) >= 2:
            cluster_id = f"{parent}_{prefix}_{num_part[:2]}"
        else:
            cluster_id = f"{parent}_{prefix}"
            
        design_to_paths[cluster_id].append(p)
        
    # Build flat image_paths and integer class labels
    image_paths = []
    labels = []
    metadata = {}
    
    for class_idx, (design_id, paths) in enumerate(sorted(design_to_paths.items())):
        for path in paths:
            image_paths.append(path)
            labels.append(class_idx)
            metadata[path] = design_id
            
    return image_paths, labels, metadata


def build_evaluation_splits(
    image_paths: List[str],
    labels: List[int],
    test_ratio: float = 0.25,
    seed: int = 42
) -> Dict[str, Dict]:
    """
    Splits the dataset into:
      1. Train Set (Seen designs for model training)
      2. Gallery Set (Reference database: 1 canon colorway per unseen test design)
      3. Query Set (Probe queries: alternate colorways/views of the test designs)
      4. Verification Pairs (Balanced set of same-design and diff-design pairs)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    class_to_paths = defaultdict(list)
    for path, lbl in zip(image_paths, labels):
        class_to_paths[lbl].append(path)
        
    all_classes = sorted(list(class_to_paths.keys()))
    num_test = max(2, int(len(all_classes) * test_ratio))
    
    shuffled_classes = list(all_classes)
    random.shuffle(shuffled_classes)
    
    test_classes = set(shuffled_classes[:num_test])
    train_classes = set(shuffled_classes[num_test:])
    
    # Training data
    train_paths, train_labels = [], []
    for cls in train_classes:
        for p in class_to_paths[cls]:
            train_paths.append(p)
            train_labels.append(cls)
            
    # Remap train labels to contiguous 0..N-1
    label_map = {orig: i for i, orig in enumerate(sorted(list(train_classes)))}
    remapped_train_labels = [label_map[lbl] for lbl in train_labels]
    
    # Test data: partition each test design into Gallery and Query
    gallery_paths, gallery_labels = [], []
    query_paths, query_labels = [], []
    
    for cls in test_classes:
        paths = class_to_paths[cls]
        # At least 1 for gallery, others for query
        gallery_paths.append(paths[0])
        gallery_labels.append(cls)
        
        if len(paths) > 1:
            for p in paths[1:]:
                query_paths.append(p)
                query_labels.append(cls)
        else:
            # If only 1 image exists, generate a synthetic colorway probe for query
            query_paths.append(paths[0])  # Will be paired with synthetic colorway transform
            query_labels.append(cls)
            
    return {
        "train": {"paths": train_paths, "labels": remapped_train_labels, "num_classes": len(train_classes)},
        "gallery": {"paths": gallery_paths, "labels": gallery_labels},
        "query": {"paths": query_paths, "labels": query_labels},
        "test_classes": list(test_classes)
    }
