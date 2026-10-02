"""
Evaluation Protocols & Rigorous Metrics for Identification & Verification.
Author: Computer Vision Engineering Team
"""

from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
import torch
import torch.nn.functional as F

def compute_similarity_matrix(query_embeddings: np.ndarray, gallery_embeddings: np.ndarray) -> np.ndarray:
    """
    Computes cosine similarity matrix between query and gallery embeddings.
    Embeddings are assumed to be L2-normalized: sim(q, g) = q . g
    Shape: [num_queries, num_gallery]
    """
    return np.matmul(query_embeddings, gallery_embeddings.T)


def evaluate_identification(
    query_embeddings: np.ndarray,
    query_labels: np.ndarray,
    gallery_embeddings: np.ndarray,
    gallery_labels: np.ndarray,
    topk_list: Tuple[int, ...] = (1, 5, 10)
) -> Dict[str, float]:
    """
    Evaluates 1:N Gallery Identification (Retrieval).
    Computes:
      - CMC (Cumulative Match Characteristic) at Rank-1, Rank-5, Rank-10
      - Mean Average Precision (mAP)
    """
    sim_matrix = compute_similarity_matrix(query_embeddings, gallery_embeddings)
    num_queries = len(query_labels)
    
    # Sort gallery items per query by descending similarity
    ranked_indices = np.argsort(-sim_matrix, axis=1)
    
    cmc_hits = {k: 0 for k in topk_list}
    average_precisions = []

    for q_idx in range(num_queries):
        target_label = query_labels[q_idx]
        ranked_gallery_labels = gallery_labels[ranked_indices[q_idx]]
        
        # Binary matches: 1 if gallery item has same design, else 0
        matches = (ranked_gallery_labels == target_label).astype(np.int32)
        total_positives = np.sum(matches)
        
        if total_positives == 0:
            continue
            
        # CMC evaluation
        for k in topk_list:
            if np.sum(matches[:k]) > 0:
                cmc_hits[k] += 1
                
        # Average Precision (AP) calculation
        cumulative_hits = np.cumsum(matches)
        ranks = np.arange(1, len(matches) + 1)
        precisions = cumulative_hits / ranks
        ap = np.sum(precisions * matches) / total_positives
        average_precisions.append(ap)

    results = {
        f"Rank-{k}_Accuracy": (cmc_hits[k] / num_queries) * 100.0 for k in topk_list
    }
    results["mAP"] = float(np.mean(average_precisions) * 100.0) if average_precisions else 0.0
    return results


def evaluate_verification(
    pair_similarities: np.ndarray,
    pair_labels: np.ndarray
) -> Dict[str, float]:
    """
    Evaluates 1:1 Pairwise Verification.
    Computes:
      - ROC Curve & ROC-AUC
      - Equal Error Rate (EER)
      - Optimal Threshold, Precision, Recall, and Accuracy
    """
    fpr, tpr, thresholds = roc_curve(pair_labels, pair_similarities)
    roc_auc = auc(fpr, tpr)
    
    # Calculate Equal Error Rate (EER): where False Accept Rate == False Reject Rate (1 - TPR)
    fnr = 1.0 - tpr
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[eer_idx]
    eer_threshold = thresholds[eer_idx]
    
    # Find threshold that maximizes F1 score
    precisions, recalls, pr_thresholds = precision_recall_curve(pair_labels, pair_similarities)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = pr_thresholds[min(best_f1_idx, len(pr_thresholds) - 1)]
    best_f1 = f1_scores[best_f1_idx]
    
    # Accuracy at optimal threshold
    predictions = (pair_similarities >= best_threshold).astype(int)
    accuracy = np.mean(predictions == pair_labels) * 100.0
    
    return {
        "ROC_AUC": float(roc_auc * 100.0),
        "EER": float(eer * 100.0),
        "EER_Threshold": float(eer_threshold),
        "Optimal_Threshold": float(best_threshold),
        "Best_F1": float(best_f1 * 100.0),
        "Accuracy_at_Optimal_Threshold": float(accuracy),
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds
    }
