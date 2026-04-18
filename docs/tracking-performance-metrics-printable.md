# GateTracker: tracking metrics reference

**Purpose:** On-paper reference for what each tracking-related metric means, how it is computed, and how to interpret trends.  
**Scope:** Ground-truth TAP-style metrics, self-supervised pairwise and temporal proxies, visibility regularization (`loss_vis`), and refinement diagnostics that affect tracking indirectly.

---

## Table of contents

1. [Visibility regularization: `loss_vis`](#1-visibility-regularization-loss_vis)
2. [Ground-truth tracking (TAP-Vid style)](#2-ground-truth-tracking-tap-vid-style)
3. [Pair-frame self-supervised proxies](#3-pair-frame-self-supervised-proxies)
4. [Multi-frame temporal self-supervised](#4-multi-frame-temporal-self-supervised)
5. [Refinement diagnostics (correspondence)](#5-refinement-diagnostics-correspondence)
6. [What to watch during training vs evaluation](#6-what-to-watch-during-training-vs-evaluation)
7. [Exporting this document to PDF](#7-exporting-this-document-to-pdf)

---

## 1. Visibility regularization (`loss_vis`)

**Where:** `visibility_regularization_loss` in `gatetracker/tracking/losses.py` (and the parallel `matching/temporal_tracking_losses.py`). Logged as `loss_vis` inside temporal tracking metrics.

**Role:** Encourages **temporally smooth** visibility probabilities and uses **entropy** to reduce **collapse** (e.g. everything always “visible”).

**Inputs:** `visibility` logits with shape `[B, Q, T]`. Internally \(p = \sigma(\text{visibility})\) with the same shape.

### Smoothness term

Let \(p_{b,q,t}\) be the sigmoid probability at batch \(b\), query \(q\), frame \(t\).

\[
\mathcal{L}_{\text{smooth}} = \mathbb{E}_{b,q,t}\,\bigl|p_{b,q,t+1} - p_{b,q,t}\bigr|
\]

(Expectation is the empirical mean over valid indices; if \(T < 2\), this term is zero.)  
**Range:** \(\mathcal{L}_{\text{smooth}} \ge 0\). Penalizes **flicker** in visibility across time.

### Entropy term (anti-collapse)

Binary entropy (with small \(\varepsilon\) for stability):

\[
H(p) = -\,p\log(p+\varepsilon) - (1-p)\log(1-p+\varepsilon).
\]

The implementation uses the **negative mean entropy**:

\[
\mathcal{L}_{\text{ent}} = -\,\mathbb{E}[H(p)].
\]

For \(p \in (0,1)\), \(H(p) > 0\), so \(\mathcal{L}_{\text{ent}} < 0\). The code comment matches this: *maximize entropy* \(\Rightarrow\) *minimize negative entropy*.

### Combined metric

\[
\mathcal{L}_{\text{vis}} = \mathcal{L}_{\text{smooth}} + 0.1\,\mathcal{L}_{\text{ent}}
= \mathcal{L}_{\text{smooth}} - 0.1\,\mathbb{E}[H(p)].
\]

This scalar is what gets logged as **`loss_vis`**. It is then scaled by `TEMPORAL_VIS_REG_WEIGHT` (default `0.1`) when added into the temporal **`loss_total`**.

### Why `loss_vis` can be negative

If temporal smoothness is small but predicted probabilities stay near **0.5** (high entropy), then \(\mathbb{E}[H(p)]\) is near its maximum \(\log 2\). The entropy branch contributes about \(-0.1\log 2 \approx -0.069\) **even when** \(\mathcal{L}_{\text{smooth}} = 0\). So **a negative `loss_vis` is not a bug**.

### Should it go to zero? Become less negative?

- **There is no correct target of zero.** \(\mathcal{L}_{\text{vis}} = 0\) would require a particular balance between flicker and mean entropy; it is **not** defined as the optimum.
- **More negative** usually means **higher** average entropy (less collapsed \(p\)) and/or **lower** frame-to-frame \(| \Delta p |\).
- **Less negative / more positive** usually means **lower** entropy (sharper 0/1 predictions) and/or **more** visibility flicker.

Judge training using **`loss_total`** and downstream **TAP metrics / visuals**, not by forcing `loss_vis` toward zero.

---

## 2. Ground-truth tracking (TAP-Vid style)

These metrics require **annotated** trajectories and visibility. Implemented in `gatetracker/tracking/metrics.py` (same logic as `matching/tracking_metrics.py`), used from `Engine.run_tracking_evaluation`, `Engine._tracking_validate_stereomis_gt`, and `test_stereomis_p3.py`.

**Tensor shapes:** `pred`, `gt`: `[N, T, 2]`; `gt_vis`, `pred_vis`: `[N, T]` (boolean after conversion).

**Default thresholds:** \(\Theta = \{1, 2, 4, 8, 16\}\) pixels.

### \(\delta_{\text{avg}}\) (`delta_avg`)

**Meaning:** Average, over \(\theta \in \Theta\), of the fraction of **GT-visible** space–time pairs whose predicted position is within \(\theta\) pixels of GT (Euclidean).

\[
\delta^{\theta} = \frac{1}{|\mathcal{V}|} \sum_{(i,t) \in \mathcal{V}} \mathbf{1}\big[\|\hat{p}_i^t - p_i^t\|_2 < \theta\big], \qquad
\mathcal{V} = \{(i,t) : \text{GT visible}\},
\]
\[
\delta_{\text{avg}} = \frac{1}{|\Theta|}\sum_{\theta \in \Theta} \delta^{\theta}.
\]

**Desideratum:** **Higher** is better (upper bound 1). **Trend:** upward over training indicates better **localization on visible points**.

### OA — occlusion accuracy

**Meaning:** Classification accuracy of predicted visibility vs GT visibility (per \((i,t)\)).

**Desideratum:** **Higher** (max 1). **Trend:** upward means better **occlusion / visibility** agreement.

### AJ — average Jaccard

**Meaning:** For each \(\theta\), define TP/FP/FN from predicted visibility, GT visibility, and whether the position error is below \(\theta\) when relevant. Jaccard is \(\mathrm{TP} / (\mathrm{TP} + \mathrm{FP} + \mathrm{FN})\) per \(\theta\); **AJ** is the mean over \(\Theta\).

**Desideratum:** **Higher** (max 1). **Trend:** upward is the standard **combined** measure (wrong visibility and large errors both hurt).

**Typical log keys:** `Test/tracking/{sequence}/delta_avg`, `.../OA`, `.../AJ`, and aggregates like `Test/tracking/mean_delta_avg`, etc.

---

## 3. Pair-frame self-supervised proxies

Computed in `compute_pairwise_tracking_losses` (`gatetracker/tracking/losses.py`; legacy twin in `matching/tracking_losses.py`). Uses a jittered grid of queries `[B, Q, 2]`, forward `track_points`, then backward tracking for cycle consistency. Logged under `{Training|Validation}/tracking/*`.

These measure **internal consistency** of the matcher’s tracker, **not** benchmark accuracy.

| Name | Meaning (summary) | Good direction |
|------|---------------------|----------------|
| `loss_cycle` | Mean \(\ell_1\) round-trip error \(\|q - p_{\text{round}}\|_1\) over \(x,y\), optionally weighted by min forward/backward visibility. | Lower |
| `loss_desc` | Mean \((1 - \cos\langle e_{\text{src}}(q), e_{\text{tgt}}(\hat{p})\rangle)\) times detached confidence clamped to \([0,1]\). | Lower |
| `loss_reg` | Mean \(\mathrm{ReLU}(\|\hat{p} - p_{\text{coarse}}\|_2 - \text{margin})\); penalizes excessive coarse→fine displacement. | Lower (often \(\approx 0\) if under margin) |
| `loss_total` | `TRACKING_LOSS_WEIGHT × (w_cycle·loss_cycle + w_desc·loss_desc + w_reg·loss_reg)`. | Lower (training objective) |
| `cycle_error` | **Unweighted** diagnostic mean \(\|q - p_{\text{round}}\|_1\). | Lower |
| `coarse_to_fine_delta` | Mean \(\|\hat{p} - p_{\text{coarse}}\|_2\) in pixels. | Contextual; spikes may be suspicious |
| `confidence_mean`, `confidence_std` | Statistics of `[B, Q]` confidence scores. | Interpret with cycle/desc; high mean alone is insufficient |
| `visibility_ratio` | Fraction of points with \(\sigma(\text{logit}) > 0.5\) (if visibility exists). | No single optimum; degenerate all-on/all-off is bad |
| `descriptor_sim` | From `compute_pairwise_tracking_metrics`: mean cosine similarity between embeddings at query vs tracked point. | Higher (≤ 1) |

**Caveat:** Improving these does **not** guarantee better \(\delta_{\text{avg}}\) / AJ on real annotations.

---

## 4. Multi-frame temporal self-supervised

From `compute_temporal_tracking_losses` in `gatetracker/tracking/losses.py`. Typical tensors: `tracks` `[B,Q,T,2]`, `visibility` `[B,Q,T]`, descriptor maps per frame, fine features `[B,T,C_f,H_f,W_f]`.

| Name | Meaning (summary) | Good direction |
|------|---------------------|----------------|
| `loss_cycle` | Forward vs backward cycle and midpoint consistency; weighted by \(\sigma(\text{vis})\) at \(t{=}0\) and optional `self_sup_mask`. | Lower |
| `loss_smooth` | Mean \(\|p_{t+1} - 2p_t + p_{t-1}\|_2\) over valid triples (second-difference / acceleration prior), with visibility and mask weights. | Lower |
| `loss_desc` | Mean \(1 - \cos\) between coarse descriptor at \(t{=}0\) and descriptors sampled along the track over \(t\). | Lower |
| `loss_feat` | Mean \(\ell_2\) distance between fine feature at query (\(t{=}0\)) and fine feature at tracked locations over \(t\). | Lower |
| `loss_vis` | Smoothness of \(\sigma(\text{visibility})\) plus weighted **negative** mean entropy (Section 1). | Signed; interpret as in Section 1 |
| `loss_total` | Weighted combination of core losses and visibility reg (see config keys in source). | Lower for optimization |

### Optional pseudo–ground-truth (temporal training)

When enabled in `TemporalTracker.training_step`, the metrics dict may also include:

| Name | Meaning | Good direction |
|------|---------|----------------|
| `loss_sup_pos` | Masked Huber on \(\|\hat{p} - p^{\text{gt}}\|_2\) where the composite supervision mask is active. | Lower |
| `loss_sup_vis` | BCE between visibility logits and `vis_target`. | Lower |
| `loss_sup_total` | Weighted sum of supervised position and visibility losses. | Lower |
| `sup_mask_fraction` | Mean supervision mask — how much pseudo-GT is applied. | Diagnostic only |
| `pseudo_lambda` | Curriculum blend factor between self-supervised total and supervised total. | Diagnostic |
| `pseudo_gt_active` | `1` if pseudo-GT branch ran, else `0`. | Diagnostic |

---

## 5. Refinement diagnostics (correspondence)

From `refinement_metrics` in `gatetracker/matching/metrics.py`, merged with stereo **matching** batch metrics. They describe **per-match** coarse vs refined targets vs GT target pixels when GT exists.

Examples: `RefinementActiveFraction`, `RefinementOffsetMean`, `RefinementScoreMean`, `CoarseErrorMean`, `RefinedErrorMean`, `RefinementGainPx`, `RefinementGainRatio`, `RefinementWinRate`, `RefinementGainConfidenceCorr`.

**When GT targets exist:** prefer **`RefinedErrorMean` < `CoarseErrorMean`**, **`RefinementGainPx` > 0**, **`RefinementWinRate`** high. These **indirectly** affect `track_points` but are **not** full-sequence trajectory metrics.

---

## 6. What to watch during training vs evaluation

1. **True tracking quality:** **`delta_avg`**, **`OA`**, **`AJ`** on GT sequences (test and optional `TRACKING_STEREOMIS_VAL`). All should **trend upward**; \(\delta_{\text{avg}}\) near 1 is excellent but often hard on difficult surgical video.

2. **Training without dense trajectory GT:** pairwise and temporal **`loss_*`** and **`cycle_error`** — generally want **downward** trends on terms that appear in the weighted objective. Treat **`loss_vis`** and **`coarse_to_fine_delta`** as diagnostics, not “minimize to zero” goals.

3. **Appearance consistency:** **`descriptor_sim` up** and **`loss_desc` down** align with stronger descriptor agreement along tracks; still **not** a substitute for TAP metrics.

---

## 7. Exporting this document to PDF

This file is plain **Markdown** (`.md`). On the cluster used for this repo, a PDF engine such as **Pandoc** was not available in the environment.

**Options:**

- **VS Code / Cursor:** Open this file → *Markdown: Open Preview* (or built-in preview) → **Print** → choose **Save as PDF** / a physical printer.
- **Browser:** Paste or open rendered HTML, then Print → Save as PDF.
- **Pandoc (if you install it):**  
  `pandoc docs/tracking-performance-metrics-printable.md -o tracking-metrics.pdf`

For double-sided printing, enable “print on both sides” in the system print dialog if your printer supports it.

---

*Document generated for the GateTracker repository; metric definitions follow the implementation as of the documentation date.*
