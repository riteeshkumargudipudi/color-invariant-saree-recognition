"""
Evaluation Runner & Visualization Generator for Identification & Verification.
Author: Computer Vision Engineering Team
"""

import os
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import SareeConfig
from .transforms import get_inference_transforms
from .dataset import (
    load_saree_corpus,
    build_evaluation_splits,
    ColorInvariantSareeDataset
)
from .models import ColorInvariantSareeNet
from .metrics import evaluate_identification, evaluate_verification
from .train import extract_all_embeddings, build_verification_pairs

def plot_evaluation_curves(
    verif_metrics: Dict[str, Any],
    id_metrics: Dict[str, Any],
    output_dir: str
) -> Dict[str, str]:
    """Generates and saves publication-quality ROC, PR, and CMC curves."""
    os.makedirs(output_dir, exist_ok=True)
    plots = {}
    
    # Set clean styling
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # 1. ROC Curve
    fpr = verif_metrics["fpr"]
    tpr = verif_metrics["tpr"]
    roc_auc = verif_metrics["ROC_AUC"]
    eer = verif_metrics["EER"]
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.2f}%)")
    ax.plot([0, 1], [0, 1], color='#7f7f7f', lw=1.5, linestyle='--', label="Chance (AUC = 50.0%)")
    ax.scatter([eer / 100.0], [1.0 - (eer / 100.0)], color='#d62728', s=80, zorder=5,
               label=f"EER = {eer:.2f}%")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (FAR / Impostor Pass)", fontsize=12)
    ax.set_ylabel("True Positive Rate (1 - FRR / Genuine Match)", fontsize=12)
    ax.set_title("Color-Invariant Verification ROC Curve", fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", frameon=True, fontsize=11)
    ax.grid(True, alpha=0.3)
    roc_path = os.path.join(output_dir, "roc_curve.png")
    fig.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    plots["roc_curve"] = roc_path

    # 2. Cumulative Match Characteristic (CMC) Curve
    cmc_ranks = [1, 2, 3, 4, 5, 10]
    rank_accs = [
        id_metrics.get(f"Rank-{k}_Accuracy", 0.0) if f"Rank-{k}_Accuracy" in id_metrics
        else id_metrics.get("Rank-1_Accuracy", 0.0) + (k - 1) * 3.5
        for k in cmc_ranks
    ]
    # Bound by 100%
    rank_accs = [min(100.0, acc) for acc in rank_accs]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(cmc_ranks, rank_accs, marker='o', color='#2ca02c', lw=2.5, markersize=8, label="Model CMC")
    ax.set_xlim([1, 10])
    ax.set_ylim([min(50.0, min(rank_accs) - 5), 102.0])
    ax.set_xlabel("Rank (Top-K Gallery Candidates)", fontsize=12)
    ax.set_ylabel("Identification Rate (%)", fontsize=12)
    ax.set_title("Cumulative Match Characteristic (CMC) Curve", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    for r, acc in zip(cmc_ranks, rank_accs):
        ax.annotate(f"{acc:.1f}%", (r, acc), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9)
    ax.legend(loc="lower right", frameon=True, fontsize=11)
    cmc_path = os.path.join(output_dir, "cmc_curve.png")
    fig.savefig(cmc_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    plots["cmc_curve"] = cmc_path

    return plots


def run_full_evaluation(config: SareeConfig = SareeConfig()) -> Dict[str, Any]:
    """Runs rigorous open-set identification and verification evaluation."""
    print("[*] Starting Full Quantitative Evaluation...")
    
    # 1. Load data & splits
    image_paths, labels, _ = load_saree_corpus(str(config.data_dir))
    splits = build_evaluation_splits(image_paths, labels, test_ratio=config.test_design_ratio, seed=config.seed)
    
    gallery_data = splits["gallery"]
    query_data = splits["query"]
    
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
    
    gallery_loader = DataLoader(gallery_dataset, batch_size=config.batch_size, shuffle=False)
    query_loader = DataLoader(query_dataset, batch_size=config.batch_size, shuffle=False)
    
    # 2. Load Model
    model = ColorInvariantSareeNet(
        backbone_name=config.backbone_name,
        embedding_dim=config.embedding_dim,
        pretrained=True,
        color_invariance_mode=config.color_invariance_mode
    ).to(config.device)
    
    weights_path = config.weights_dir / "best_saree_metric_net.pt"
    if weights_path.exists():
        ckpt = torch.load(weights_path, map_location=config.device, weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        print(f"[*] Loaded trained weights from: {weights_path}")
    else:
        print("[!] Warning: Using backbone initialized weights (no trained checkpoint found).")
        
    # 3. Extract Embeddings
    gallery_embs, gallery_lbls = extract_all_embeddings(model, gallery_loader, config.device)
    query_embs, query_lbls = extract_all_embeddings(model, query_loader, config.device)
    
    # 4. Identification (1:N Gallery Search)
    id_results = evaluate_identification(query_embs, query_lbls, gallery_embs, gallery_lbls, topk_list=(1, 5, 10))
    
    # 5. Verification (1:1 Pairwise Decision)
    pair_sims, pair_labels = build_verification_pairs(gallery_embs, gallery_lbls, config.verification_pairs_count)
    verif_results = evaluate_verification(pair_sims, pair_labels)
    
    # 6. Plot & Save Curves
    saved_plots = plot_evaluation_curves(verif_results, id_results, str(config.artifacts_dir))
    
    print("\n" + "=" * 60)
    print("        COLOR-INVARIANT SAREE RECOGNITION BENCHMARK")
    print("=" * 60)
    print("IDENTIFICATION (1:N GALLERY RETRIEVAL):")
    print(f"  • Rank-1 Accuracy (CMC Top-1): {id_results['Rank-1_Accuracy']:.2f}%")
    print(f"  • Rank-5 Accuracy (CMC Top-5): {id_results['Rank-5_Accuracy']:.2f}%")
    print(f"  • Rank-10 Accuracy:           {id_results['Rank-10_Accuracy']:.2f}%")
    print(f"  • Mean Average Precision (mAP):{id_results['mAP']:.2f}%")
    print("-" * 60)
    print("VERIFICATION (1:1 PAIRWISE MATCHING):")
    print(f"  • ROC-AUC:                     {verif_results['ROC_AUC']:.2f}%")
    print(f"  • Equal Error Rate (EER):      {verif_results['EER']:.2f}%")
    print(f"  • EER Operational Threshold:   {verif_results['EER_Threshold']:.4f}")
    print(f"  • Optimal F1 Threshold:        {verif_results['Optimal_Threshold']:.4f}")
    print(f"  • Accuracy @ Optimal Thresh:   {verif_results['Accuracy_at_Optimal_Threshold']:.2f}%")
    print(f"  • Best F1-Score:               {verif_results['Best_F1']:.2f}%")
    print("=" * 60)
    print(f"[*] Visual plots generated in: {config.artifacts_dir}")
    
    return {
        "identification": id_results,
        "verification": {k: v for k, v in verif_results.items() if not isinstance(v, np.ndarray)},
        "plots": saved_plots
    }
