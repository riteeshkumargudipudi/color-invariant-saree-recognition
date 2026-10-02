# Approach Note: Color-Invariant Saree Design Recognition

---

### Option A: Concise Submission Text (Strictly Under 500 Characters)
*(Character count: 489 characters including spaces)*

> We address textile design retrieval using EfficientNet-B0 with Generalized Mean (GeM, p=3) pooling and a 256-D L2-normalized hyperspherical bottleneck. Standard CNNs overfit to color histograms; our pipeline enforces color invariance via Sobel spatial gradients, random channel shuffling, and multi-scale chromatic jitter. We train using combined ArcFace (s=32, m=0.35) and Batch-Hard Triplet loss with a balanced (P=8, K=4) sampler, mapping motifs to an angular metric space for sub-millisecond MIPS cosine search.

---

### Option B: Extended Technical Note (~500 Words)

#### 1. Architectural Choice & Rationale
Standard deep convolutional backbones pre-trained on ImageNet suffer from severe texture and color bias (Geirhos et al., 2018). In textile retrieval, two sarees featuring identical motifs (e.g., Paisley/Kalka, Paithani peacock, or Patola lattice) frequently appear in radically distinct colorways (e.g., Crimson/Gold vs. Teal/Silver). A naive CNN matches images based on global color distribution rather than motif geometry.

To eliminate this bias without prohibitive computational cost, we propose **ColorInvariantSareeNet**, an edge-deployable deep metric learning system:
* **Backbone**: **EfficientNet-B0** (3.9M parameters, 0.39 GFLOPs). It provides superior receptive-field scalability with inverted residual mobile bottlenecks (MBConv), outperforming bloated Vision Transformers that easily overfit on specialized textile sets.
* **Salient Feature Pooling**: We replace standard Global Average Pooling (GAP) with **Generalized Mean (GeM) Pooling** with a learnable power parameter $p$ (initialized at $p=3.0$). Unlike GAP, which dilutes fine recurring motifs across plain fabric folds, GeM dynamically sharpens attention towards prominent geometric contours.
* **Hyperspherical Bottleneck**: A linear projection maps features into a compact 256-D metric space, normalized via $\ell_2$-normalization onto a unit hypersphere $\mathbb{S}^{255}$.

#### 2. Pre- and Post-Processing Pipeline
* **Preprocessing & Disentanglement**:
  1. Input images are resized to $224 \times 224$.
  2. Chromatic and structural representations are decoupled. In our hybrid structural mode, spatial Sobel gradient magnitudes ($G = \sqrt{G_x^2 + G_y^2}$) and high-pass spatial residuals are extracted to isolate high-frequency zari thread boundaries and block-print silhouettes, completely removing background dye dependencies.
* **Post-Processing & Retrieval**:
  1. Embeddings are $\ell_2$-normalized: $\|\hat{e}\|_2 = 1.0$.
  2. Distance computation simplifies to dot-product cosine similarity $\cos(\theta) = \hat{e}_q^\top \hat{e}_g$, enabling $\mathcal{O}(d)$ sub-millisecond Maximum Inner Product Search (MIPS) across large gallery catalogs.

#### 3. Training Strategy & Loss Formulations
* **Metric Learning Objective**: We combine **ArcFace (Additive Angular Margin Loss)** and **Batch-Hard Triplet Margin Loss**:
  $$\mathcal{L} = \mathcal{L}_{\text{ArcFace}}(s=32, m=0.35) + \lambda \mathcal{L}_{\text{Triplet}}(m=0.30)$$
  ArcFace enforces an angular margin $m$ between different motifs on the hypersphere, while batch-hard triplet mining acts as a color-adversarial regularizer: if two distinct designs share identical color palettes, the triplet loss penalizes the model for clustering them together.
* **Sampling**: A **Balanced Batch Sampler** ($P=8$ classes $\times$ $K=4$ colorway instances = batch size 32) ensures each mini-batch contains both multiple positive colorway pairs and hard negative distractors.
* **Augmentations**: Aggressive chromatic perturbation (Random Channel Permutation, Color Jitter with hue wrap $\pm 0.5$, Random Grayscale $p=0.3$, Random Autocontrast, Random Erasing).
