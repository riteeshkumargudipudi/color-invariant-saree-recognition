"""
Production Inference Engine: 1:N Gallery Identification and 1:1 Pair Verification.
Author: Computer Vision Engineering Team
"""

import os
from typing import List, Dict, Any, Tuple, Union, Optional
from PIL import Image
import numpy as np
import torch
import torch.nn as nn

from .models import ColorInvariantSareeNet
from .transforms import get_inference_transforms

class SareeRecognizer:
    """
    Color-Invariant Saree Recognition and Verification Engine.
    
    Provides:
      - 1:N Gallery Identification (returns top-k design matches)
      - 1:1 Pairwise Verification (decides whether two sarees carry the same motif)
    """
    def __init__(
        self,
        weights_path: Optional[str] = None,
        backbone_name: str = "efficientnet_b0",
        embedding_dim: int = 256,
        color_invariance_mode: str = "hybrid_invariant",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        verification_threshold: float = 0.65
    ):
        self.device = device
        self.verification_threshold = verification_threshold
        self.transforms = get_inference_transforms()
        
        # Initialize model
        self.model = ColorInvariantSareeNet(
            backbone_name=backbone_name,
            embedding_dim=embedding_dim,
            pretrained=True if weights_path is None else False,
            color_invariance_mode=color_invariance_mode
        )
        
        if weights_path and os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            self.model.load_state_dict(state_dict)
            print(f"[*] Loaded trained model weights from: {weights_path}")
            
        self.model.to(self.device)
        self.model.eval()
        
        # Gallery index cache
        self.gallery_embeddings: Optional[np.ndarray] = None
        self.gallery_paths: List[str] = []
        self.gallery_labels: List[Any] = []

    def extract_embedding(self, image: Union[str, Image.Image]) -> np.ndarray:
        """Extracts an L2-normalized 256-D embedding from an image or path."""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")
            
        tensor = self.transforms(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            emb = self.model(tensor)
            
        return emb.cpu().squeeze(0).numpy()

    def index_gallery(self, image_paths: List[str], labels: List[Any]) -> None:
        """Indexes reference gallery database into memory for fast cosine search."""
        self.gallery_paths = image_paths
        self.gallery_labels = labels
        embeddings = []
        
        print(f"[*] Indexing {len(image_paths)} reference saree designs into gallery...")
        for p in image_paths:
            emb = self.extract_embedding(p)
            embeddings.append(emb)
            
        self.gallery_embeddings = np.array(embeddings)
        print(f"[*] Gallery index ready: shape {self.gallery_embeddings.shape}")

    def identify(self, query_image: Union[str, Image.Image], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        1:N Identification: Ranks reference gallery by similarity to the query.
        Returns top-k best matches with similarity scores and match verification status.
        """
        if self.gallery_embeddings is None:
            raise ValueError("Gallery must be indexed with index_gallery() before calling identify().")
            
        q_emb = self.extract_embedding(query_image)
        # Cosine similarity via inner product on normalized embeddings
        scores = np.dot(self.gallery_embeddings, q_emb)
        ranked_indices = np.argsort(-scores)[:top_k]
        
        results = []
        for rank, idx in enumerate(ranked_indices, start=1):
            score = float(scores[idx])
            results.append({
                "rank": rank,
                "gallery_path": self.gallery_paths[idx],
                "design_label": self.gallery_labels[idx],
                "cosine_similarity": round(score, 4),
                "is_confident_match": score >= self.verification_threshold
            })
            
        return results

    def verify(
        self,
        image_a: Union[str, Image.Image],
        image_b: Union[str, Image.Image],
        custom_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        1:1 Verification: Decides whether two images carry the same surface design,
        invariant to colorway differences.
        """
        threshold = custom_threshold if custom_threshold is not None else self.verification_threshold
        
        emb_a = self.extract_embedding(image_a)
        emb_b = self.extract_embedding(image_b)
        
        cosine_sim = float(np.dot(emb_a, emb_b))
        is_same_design = cosine_sim >= threshold
        
        # Confidence score scaled to [0, 1] relative to threshold
        confidence = round(float(np.clip((cosine_sim + 1.0) / 2.0, 0.0, 1.0)), 4)
        
        return {
            "is_same_design": is_same_design,
            "cosine_similarity": round(cosine_sim, 4),
            "threshold": threshold,
            "confidence": confidence
        }
