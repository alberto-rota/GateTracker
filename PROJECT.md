# GateTracker: Self-Supervised Dense Matching and Long-Term Tracking for Endoscopic Video

## 1. Abstract

We present **GateTracker**, a self-supervised framework for dense feature matching and long-term point tracking in endoscopic and laparoscopic video. By leveraging monocular depth estimation and relative camera pose to synthesize novel views, we generate pseudo ground-truth correspondences that supervise descriptor learning without manual annotation. Our method introduces an adaptive gated hierarchical fusion module over frozen DINOv3 features, a masked InfoNCE objective that accounts for occlusion and photometric confidence, and a coarse-to-fine matching pipeline that achieves sub-patch precision. For temporal tracking, we train a TAPIR-inspired refinement network with multi-frame self-supervised losses, enabling accurate long-term dense point tracking across challenging deformable surgical scenes.

---

## 2. Method Overview

### 2.1 Self-Supervised Pseudo Ground Truth via Novel View Synthesis

Ground-truth correspondences are scarce in surgical video. We construct pseudo-GT by warping source frames to novel viewpoints using predicted geometry. Given a frame pair \((I_s, I_t)\), we estimate monocular depth \(\hat{D}_s\) via MoGe and recover relative camera pose \(T_{s \to t} \in SE(3)\). A target view is synthesized by projecting source pixels into 3D and re-projecting under \(T_{s \to t}\):

\[
\hat{I}_t(u') = I_s(u), \quad u' = \pi\!\left(K \, T_{s \to t} \, \hat{D}_s(u) \, \pi^{-1}(u, K)\right)
\]

where \(\pi\) denotes perspective projection and \(K\) the intrinsic matrix. Depth is rescaled as \(D = (\hat{D}^{-1}) \cdot \alpha + \beta\) with learnable scale \(\alpha\) and bias \(\beta\). The warped image \(\hat{I}_t\) and the dense pixel-level correspondence map \(u \mapsto u'\) serve as pseudo ground truth.

**Confidence masking.** Not all warped pixels are reliable—occlusions, depth errors, and specular reflections introduce noise. We compute a per-pixel confidence mask by combining: (i) forward-backward depth consistency, (ii) photometric similarity between \(\hat{I}_t\) and \(I_t\), and (iii) valid projection bounds. This mask is aggregated to the patch level for use in the contrastive objective.

### 2.2 Adaptive Gated Hierarchical Fusion over DINOv3

Vision transformers encode different levels of abstraction at different layers—shallow layers capture fine textures while deep layers encode semantic structure. Rather than using a single layer or a fixed combination, we introduce **Register-Gated Hierarchical Fusion** that learns input-dependent layer weights.

Given hidden states \(\{h_\ell\}_{\ell \in \mathcal{L}}\) from selected DINOv3 transformer layers, each is projected to a shared dimension \(d\):

\[
\hat{h}_\ell = W_\ell \, h_\ell + b_\ell
\]

Two complementary gating signals are computed per layer:

- **Local gate** \(g_\ell^{\text{loc}}(i)\): A linear projection of the patch token at spatial position \(i\), capturing position-specific layer preference.
- **Register gate** \(g_\ell^{\text{reg}}\): A linear projection of the mean register token, providing global scene-level context.

The combined logit is:

\[
z_\ell(i) = \tanh\!\big(g_\ell^{\text{loc}}(i) + g_\ell^{\text{reg}}\big) \cdot B
\]

where \(B\) is a learned scaling parameter. Logits are mean-centered across layers per position for stability, then converted to weights via temperature-scaled softmax:

\[
w_\ell(i) = \text{softmax}_\ell\!\left(\frac{z_\ell(i) - \bar{z}(i)}{\tau}\right)
\]

An optional uniform mixing coefficient \(\lambda\) prevents mode collapse: \(\tilde{w}_\ell = (1 - \lambda)\,w_\ell + \lambda / |\mathcal{L}|\). During training, stochastic layer dropout randomly zeros out individual layer contributions. The fused representation is:

\[
f(i) = \text{MLP}\!\left(\sum_{\ell \in \mathcal{L}} \tilde{w}_\ell(i) \, \hat{h}_\ell(i)\right)
\]

The DINOv3 backbone remains **frozen**; only the fusion gates, projections, and output MLP are trained.

### 2.3 Coarse-to-Fine Feature Matching

Matching proceeds in two stages:

**Coarse stage.** Given fused descriptors \(F_s, F_t \in \mathbb{R}^{d \times N}\) for source and target, we compute the cosine similarity matrix \(S = F_s^\top F_t / \|F_s\| \|F_t\|\) and extract mutual nearest neighbors as coarse correspondences at the patch-grid resolution (stride 14).

**Fine stage.** A lightweight **Local Refinement Feature Head** produces dense feature maps at stride 4 by fusing RGB input, interpolated coarse descriptors, and a context layer from the backbone:

\[
F^{\text{fine}} = h_\theta\!\big(\text{concat}(I_{\downarrow 4},\; \text{upsample}(F^{\text{coarse}}),\; C)\big) \in \mathbb{R}^{B \times d_f \times H/4 \times W/4}
\]

Sub-patch refinement is performed via **feature soft-argmax**: a local window \(W_r\) around each coarse match is extracted from \(F_t^{\text{fine}}\), dot-product similarities with the query descriptor are computed, and the expected position is obtained as:

\[
\hat{u}^{\text{fine}} = \sum_{v \in W_r} v \cdot \text{softmax}\!\big(F_s^{\text{fine}}(u)^\top F_t^{\text{fine}}(v) / \tau_r\big)
\]

An alternative FFT-based phase-correlation refiner operates directly on RGB patches for geometry-only refinement.

### 2.4 Self-Supervised Descriptor Learning with Masked InfoNCE

Descriptors are trained with a **masked symmetric InfoNCE** loss. For each source patch \(i\), the pseudo-GT warp identifies the positive target patch \(y_i\). The logit matrix is:

\[
L_{ij} = \frac{f_s(i)^\top f_t(j)}{\tau_c}
\]

The confidence mask \(M\) (Section 2.1) determines which negative pairs are valid—unreliable patches are excluded from the denominator while positives are always retained:

\[
\tilde{L}_{ij} = \begin{cases} L_{ij} & \text{if } M_j = 1 \text{ or } j = y_i \\ -\infty & \text{otherwise} \end{cases}
\]

The loss for direction \(s \to t\) is:

\[
\mathcal{L}^{s \to t}_{\text{InfoNCE}} = -\frac{1}{|\mathcal{A}|} \sum_{i \in \mathcal{A}} \log \frac{\exp(\tilde{L}_{i,y_i})}{\sum_j \exp(\tilde{L}_{ij})}
\]

where \(\mathcal{A}\) is the set of anchor patches with valid positives. The symmetric variant averages both directions:

\[
\mathcal{L}_{\text{desc}} = \frac{1}{2}\!\left(\mathcal{L}^{s \to t}_{\text{InfoNCE}} + \mathcal{L}^{t \to s}_{\text{InfoNCE}}\right)
\]

Triplet mining selects anchors on a regular grid, with positives from the warp map and hard negatives from the similarity matrix, ensuring diverse and informative training signal.

### 2.5 Long-Term Dense Point Tracking

For temporal tracking beyond frame pairs, we employ a **Temporal Refinement Network** inspired by TAPIR. Given a query point \(p_q\) in frame \(t_0\), coarse correspondences are first established in each target frame via the matching pipeline (Section 2.3). The refinement network then iteratively updates tracks using:

1. **Local correlation volumes**: For each tracked point, a local window of target features is correlated with the query descriptor.
2. **Fourier position encoding**: The current estimate \((\hat{x}, \hat{y}, t)\) is encoded with sinusoidal features to provide continuous spatial-temporal context.
3. **Depthwise temporal convolutions**: Information is propagated across the time axis through weight-shared temporal conv blocks, enabling the network to reason about motion coherence.
4. **Iterative refinement**: For \(K\) iterations (weight-shared), the network predicts residual position updates \(\Delta p^{(k)}\) and visibility logits \(v^{(k)}\):

\[
\hat{p}^{(k+1)} = \hat{p}^{(k)} + \Delta p^{(k)}, \quad \hat{v}^{(k)} = \sigma(v^{(k)}_{\text{logit}})
\]

The tracking phase is trained with self-supervised temporal losses:

- **Cycle consistency**: Forward-backward tracking should return to the query point: \(\mathcal{L}_{\text{cyc}} = \| p_q - \hat{p}_{t_0 \to t \to t_0} \|_1\)
- **Temporal smoothness**: Penalizes acceleration: \(\mathcal{L}_{\text{smooth}} = \| \hat{p}^{t+1} - 2\hat{p}^t + \hat{p}^{t-1} \|_2\)
- **Descriptor consistency**: Tracked points should maintain similar descriptors across time
- **Feature persistence**: Features at tracked locations should remain stable over the sequence
- **Visibility regularization**: Prevents degenerate all-visible or all-occluded predictions

### 2.6 Multi-Dataset Curriculum Training

Surgical scenes vary dramatically in visual complexity across datasets. We employ a curriculum strategy that progressively increases task difficulty:

**Frame-skip curriculum.** Training begins with temporally adjacent frames (small motion) and gradually increases the frame gap. The skip distance is incremented every \(N_{\text{skip}}\) epochs, exposing the model to progressively larger viewpoint and deformation changes.

**Multi-dataset mixing.** The `MultiDataset` loader combines SCARED (stereo endoscopy), CHOLEC80 (monocular cholecystectomy), and StereoMIS (stereo with tracking annotations) with configurable sampling weights, ensuring the model generalizes across camera types, procedures, and tissue appearances.

---

## 3. Architecture

### 3.1 MatcherModel

`MatcherModel` extends `FeatureExtractor` and serves as the backbone for both tasks:

| Component | Description |
|---|---|
| **DINOv3 Backbone** | Frozen ViT with register tokens; `output_hidden_states=True` exposes all layer activations |
| **RegisterGatedHierarchicalFusion** | Adaptive layer fusion with local + register gates (Section 2.2) |
| **LocalRefinementFeatureHead** | Stride-4 dense features from RGB + coarse + context (Section 2.3) |
| **DPT_Predictor** | Intel DPT-style head on multi-scale DINO features for inverse-depth prediction |
| **TrackingHead** | Local correlation + soft-argmax + visibility MLP for pairwise point tracking |

Feature extraction produces patch embeddings \(F \in \mathbb{R}^{B \times d \times H_p \times W_p}\) at the ViT patch stride (14), optionally resampled to a target resolution. The model supports gated diagnostic mode, outputting per-layer weight maps for interpretability.

### 3.2 TemporalTracker

`TemporalTracker` composes a frozen `MatcherModel` with a `TemporalRefinementNetwork`:

1. **Initialization**: Coarse per-frame matches from the frozen matcher provide starting track estimates \(\hat{p}^{(0)}_t\) for all query points across \(T\) frames.
2. **Refinement**: The temporal refinement network (Section 2.5) iteratively updates all tracks jointly, leveraging temporal convolutions to enforce motion coherence.
3. **Inference**: `track_long_sequence` processes videos in overlapping sliding windows with confidence-weighted blending at boundaries.

---

## 4. Training Protocol

Training proceeds in two phases:

### Phase 1: Descriptor Pretraining

**Objective**: Learn discriminative patch descriptors via self-supervised matching.

| Aspect | Detail |
|---|---|
| **Frozen** | DINOv3 backbone |
| **Trained** | Gated fusion, refinement head, depth head |
| **Pseudo-GT** | MoGe depth + relative pose \(\to\) novel view warp |
| **Loss** | \(\mathcal{L} = \mathcal{L}_{\text{desc}} + \lambda_{\text{ref}} \mathcal{L}_{\text{refine}} + \lambda_{\text{epi}} \mathcal{L}_{\text{epi}}\) |
| **Curriculum** | Frame-skip increases every \(N_{\text{skip}}\) epochs |
| **Output** | `save_pretrained_descriptors` strips depth/tracking heads for Phase 2 |

The refinement loss \(\mathcal{L}_{\text{refine}}\) is a weighted Smooth-L1 between predicted fine correspondences and pseudo-GT pixel locations. The optional epipolar loss \(\mathcal{L}_{\text{epi}}\) penalizes correspondences that violate the epipolar constraint from the estimated fundamental matrix.

### Phase 2: Temporal Tracking

**Objective**: Learn temporally coherent long-term point tracking.

| Aspect | Detail |
|---|---|
| **Frozen** | DINOv3 backbone + pretrained fusion/refinement (from Phase 1) |
| **Trained** | TemporalRefinementNetwork |
| **Data** | Sequence windows of length \(T\) (e.g., 8–16 frames) |
| **Loss** | \(\mathcal{L} = \lambda_{\text{cyc}} \mathcal{L}_{\text{cyc}} + \lambda_{\text{sm}} \mathcal{L}_{\text{smooth}} + \lambda_{\text{dc}} \mathcal{L}_{\text{desc\_cons}} + \lambda_{\text{fp}} \mathcal{L}_{\text{persist}} + \lambda_{\text{vis}} \mathcal{L}_{\text{vis}}\) |
| **Augmentation** | Geometric (affine, thin-plate spline) + photometric (color jitter, blur) |

---

## 5. Evaluation Metrics

### 5.1 Feature Matching Metrics

| Metric | Definition |
|---|---|
| **Matching Rate** | Fraction of query points that produce a mutual nearest-neighbor match |
| **Inlier Ratio** | Fraction of matches consistent with the estimated fundamental matrix (Sampson error \(< \epsilon\)) |
| **NCM** (Num. Correct Matches) | Matches within a pixel threshold of ground truth |
| **Success Rate** | Fraction of image pairs where the inlier ratio exceeds a threshold |
| **Epipolar Error** | Mean Sampson distance of matches to the ground-truth epipolar lines |

### 5.2 Tracking Metrics (TAP-Vid Protocol)

Following the TAP-Vid benchmark, we report:

**Position accuracy** at threshold \(\theta\) (in pixels):

\[
\delta^\theta = \frac{1}{|\mathcal{V}|} \sum_{(i,t) \in \mathcal{V}} \mathbf{1}\!\left[\|\hat{p}_i^t - p_i^t\|_2 < \theta\right]
\]

**Average \(\delta\)** over thresholds \(\Theta = \{1, 2, 4, 8, 16\}\) pixels:

\[
\delta_{\text{avg}} = \frac{1}{|\Theta|} \sum_{\theta \in \Theta} \delta^\theta
\]

**Occlusion Accuracy (OA)**: Binary classification accuracy of visibility predictions.

**Average Jaccard (AJ)**: Jointly evaluates position and visibility:

\[
\text{AJ} = \frac{1}{|\Theta|} \sum_{\theta \in \Theta} \frac{\text{TP}^\theta}{\text{TP}^\theta + \text{FP}^\theta + \text{FN}^\theta}
\]

where \(\text{TP}^\theta\) counts visible points tracked within threshold \(\theta\), \(\text{FP}^\theta\) counts points predicted visible but incorrectly tracked, and \(\text{FN}^\theta\) counts points that are visible but predicted occluded.

---

## 6. Datasets

### 6.1 SCARED (Stereo Correspondence and Reconstruction of Endoscopic Data)

Structured-light stereo endoscopy dataset providing calibrated stereo pairs with dense depth maps. Used primarily for Phase 1 descriptor pretraining, as it offers reliable depth supervision for novel view synthesis.

| Property | Value |
|---|---|
| **Modality** | Stereo endoscopy (da Vinci) |
| **Scenes** | 7 sequences across multiple patients |
| **Ground truth** | Dense structured-light depth, stereo calibration |
| **Role** | Primary source for self-supervised correspondence via depth-based warping |

### 6.2 CHOLEC80

Large-scale monocular cholecystectomy video dataset. Provides diverse surgical scenes with significant visual variation (smoke, bleeding, instrument occlusion) but no ground-truth geometry.

| Property | Value |
|---|---|
| **Modality** | Monocular laparoscopy |
| **Videos** | 80 cholecystectomy procedures |
| **Ground truth** | Phase annotations only (no geometry) |
| **Role** | Domain diversity during multi-dataset training; MoGe provides estimated depth |

### 6.3 StereoMIS (Stereo Minimally Invasive Surgery)

Stereo laparoscopic dataset with annotated point tracks for tracking evaluation. Serves as the primary benchmark for Phase 2 temporal tracking.

| Property | Value |
|---|---|
| **Modality** | Stereo laparoscopy |
| **Sequences** | Multiple procedures with varying complexity |
| **Ground truth** | Sparse annotated point tracks with visibility labels |
| **Role** | Evaluation benchmark for tracking (TAP-Vid metrics); also used for sequence-based training |
