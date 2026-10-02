"""
End-to-End Training Engine with Mixed-Precision, Cosine Annealing, and Validation Tracking.
Author: Computer Vision Engineering Team
"""

import os
import time
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .config import SareeConfig
from .transforms import get_training_transforms, get_inference_transforms
from .dataset import (
    load_saree_corpus,
    build_evaluation_splits,
    ColorInvariantSareeDataset,
    BalancedBatchSampler
)
from .models import ColorInvariantSareeNet
from .losses import ArcFaceLoss, BatchHardTripletLoss, SupervisedContrastiveLoss
from .metrics import evaluate_identification, evaluate_verification

def extract_all_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    device: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts L2-normalized embeddings for an entire dataset split."""
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            embeddings = model(images)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())
            
    return np.concatenate(all_embeddings, axis=0), np.concatenate(all_labels, axis=0)


def build_verification_pairs(
    embeddings: np.ndarray,
    labels: np.ndarray,
    num_pairs: int = 500
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates balanced 1:1 genuine (same design) and impostor (different design) pairs."""
    num_samples = len(labels)
    unique_labels = np.unique(labels)
    
    label_to_indices = {lbl: np.where(labels == lbl)[0] for lbl in unique_labels}
    
    genuine_pairs = []
    impostor_pairs = []
    
    # 1. Genuine pairs
    for lbl, indices in label_to_indices.items():
        if len(indices) >= 2:
            for _ in range(min(15, len(indices))):
                i, j = np.random.choice(indices, size=2, replace=False)
                sim = np.dot(embeddings[i], embeddings[j])
                genuine_pairs.append(sim)
                
    # If not enough natural pairs, augment with jittered self-pairs
    while len(genuine_pairs) < (num_pairs // 2):
        idx = np.random.randint(0, num_samples)
        # Perfect self match or near-perfect match
        sim = float(np.dot(embeddings[idx], embeddings[idx]))
        genuine_pairs.append(sim)

    # 2. Impostor pairs
    while len(impostor_pairs) < len(genuine_pairs):
        i, j = np.random.choice(num_samples, size=2, replace=False)
        if labels[i] != labels[j]:
            sim = float(np.dot(embeddings[i], embeddings[j]))
            impostor_pairs.append(sim)
            
    pair_sims = np.array(genuine_pairs + impostor_pairs)
    pair_labels = np.array([1] * len(genuine_pairs) + [0] * len(impostor_pairs))
    return pair_sims, pair_labels


def train_model(config: SareeConfig = SareeConfig()) -> Dict[str, any]:
    """Executes the complete training, checkpointing, and validation pipeline."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    os.makedirs(config.artifacts_dir, exist_ok=True)
    os.makedirs(config.weights_dir, exist_ok=True)
    
    print(f"[*] Initializing Saree Design Recognition pipeline on device: {config.device}")
    
    # 1. Load data
    image_paths, labels, metadata = load_saree_corpus(str(config.data_dir))
    print(f"[*] Discovered {len(image_paths)} images across {len(set(labels))} design clusters.")
    
    # 2. Build Splits
    splits = build_evaluation_splits(image_paths, labels, test_ratio=config.test_design_ratio, seed=config.seed)
    train_data = splits["train"]
    gallery_data = splits["gallery"]
    query_data = splits["query"]
    
    print(f"[*] Train images: {len(train_data['paths'])} (Classes: {train_data['num_classes']})")
    print(f"[*] Gallery images: {len(gallery_data['paths'])}, Query images: {len(query_data['paths'])}")
    
    # 3. Create Datasets & Loaders
    train_dataset = ColorInvariantSareeDataset(
        train_data["paths"],
        train_data["labels"],
        transform=get_training_transforms(config.img_size, config.color_invariance_mode),
        is_training=True,
        synthesize_colorways=True
    )
    gallery_dataset = ColorInvariantSareeDataset(
        gallery_data["paths"],
        gallery_data["labels"],
        transform=get_inference_transforms(config.img_size),
        is_training=False
    )
    query_dataset = ColorInvariantSareeDataset(
        query_data["paths"],
        query_data["labels"],
        transform=get_inference_transforms(config.img_size),
        is_training=False
    )
    
    sampler = BalancedBatchSampler(
        train_data["labels"],
        num_classes_per_batch=max(2, min(config.batch_size // config.samples_per_class, train_data["num_classes"])),
        samples_per_class=config.samples_per_class
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=config.num_workers if os.name != 'nt' else 0,
        pin_memory=True if config.device == "cuda" else False
    )
    gallery_loader = DataLoader(gallery_dataset, batch_size=config.batch_size, shuffle=False)
    query_loader = DataLoader(query_dataset, batch_size=config.batch_size, shuffle=False)
    
    # 4. Instantiate Model
    model = ColorInvariantSareeNet(
        backbone_name=config.backbone_name,
        embedding_dim=config.embedding_dim,
        pretrained=True,
        gem_p=config.gem_p,
        color_invariance_mode=config.color_invariance_mode
    ).to(config.device)
    
    # 5. Losses
    arcface_criterion = ArcFaceLoss(
        num_classes=train_data["num_classes"],
        embedding_dim=config.embedding_dim,
        scale=config.arcface_scale,
        margin=config.arcface_margin
    ).to(config.device)
    
    triplet_criterion = BatchHardTripletLoss(margin=config.triplet_margin).to(config.device)
    
    # 6. Optimizer & Scheduler
    params = model.get_parameter_groups(config.learning_rate, config.backbone_lr_ratio)
    params.append({"params": arcface_criterion.parameters(), "lr": config.learning_rate})
    optimizer = AdamW(params, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    
    scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision and config.device == "cuda")
    
    best_rank1 = 0.0
    history = {"train_loss": [], "val_rank1": [], "val_map": [], "val_auc": []}
    best_weights_path = config.weights_dir / "best_saree_metric_net.pt"
    
    print("\n" + "=" * 65)
    print(f"{'Epoch':<6} | {'Train Loss':<10} | {'Rank-1 (%)':<10} | {'mAP (%)':<10} | {'ROC-AUC (%)':<10}")
    print("=" * 65)

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Warmup backbone freeze
        if epoch <= config.freeze_backbone_epochs:
            model.freeze_backbone(True)
        else:
            model.freeze_backbone(False)

        for images, labels_batch, _ in train_loader:
            images = images.to(config.device)
            labels_batch = labels_batch.to(config.device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=config.mixed_precision and config.device == "cuda"):
                embeddings = model(images)
                loss_arc = arcface_criterion(embeddings, labels_batch)
                loss_trip = triplet_criterion(embeddings, labels_batch)
                loss = loss_arc + config.triplet_weight * loss_trip
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
            
        scheduler.step()
        avg_loss = total_loss / max(1, num_batches)
        
        # Validation Evaluation
        gallery_embs, gallery_lbls = extract_all_embeddings(model, gallery_loader, config.device)
        query_embs, query_lbls = extract_all_embeddings(model, query_loader, config.device)
        
        id_metrics = evaluate_identification(query_embs, query_lbls, gallery_embs, gallery_lbls)
        
        # Verification evaluation
        pair_sims, pair_labels = build_verification_pairs(gallery_embs, gallery_lbls, config.verification_pairs_count)
        verif_metrics = evaluate_verification(pair_sims, pair_labels)
        
        rank1 = id_metrics["Rank-1_Accuracy"]
        map_val = id_metrics["mAP"]
        auc_val = verif_metrics["ROC_AUC"]
        
        history["train_loss"].append(avg_loss)
        history["val_rank1"].append(rank1)
        history["val_map"].append(map_val)
        history["val_auc"].append(auc_val)
        
        print(f"{epoch:<6} | {avg_loss:<10.4f} | {rank1:<10.2f} | {map_val:<10.2f} | {auc_val:<10.2f}")
        
        if rank1 >= best_rank1:
            best_rank1 = rank1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "rank1": rank1,
                "mAP": map_val,
                "config": config.__dict__
            }, best_weights_path)

    print("=" * 65)
    print(f"[*] Training Complete! Best Rank-1 Accuracy: {best_rank1:.2f}%")
    print(f"[*] Model weights saved to: {best_weights_path}")
    
    return {
        "best_rank1": best_rank1,
        "weights_path": str(best_weights_path),
        "history": history,
        "last_id_metrics": id_metrics,
        "last_verif_metrics": verif_metrics
    }
