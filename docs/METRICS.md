# GateTracker metrics reference

This file is split by **training phase**. Each phase section lists scalar metrics (and a few non-scalar logs), how they are computed, W&B keys, and console abbreviations where applicable.

| Phase | Scope |
|--------|--------|
| [Pretrain](#phase-pretrain) | Descriptor + matching + refinement; `gatetracker/engine.py` `run_epoch` |
| [Tracking](#phase-tracking) | Temporal refinement on frozen matcher; `tracking_trainloop` / `_run_tracking_epoch` |

Shared conventions (naming, W&B axes, console rules) are described once under [Naming](#naming-canonical-name-wb-key-console); they apply to both phases unless noted.

---

## Phase: Pretrain

This subsection describes **scalar metrics** computed and logged during **`PHASE: pretrain`**: descriptor learning with DINO-style backbones, gated hierarchical fusion, patch matching, optional fine refinement, and **no** pairwise tracking loss by default (`pairwise_tracking_enabled` is false unless `TRACKING_MODE` legacy is set).

Sources of truth in code:

- Training / validation / test loop: `gatetracker/engine.py` (`run_epoch`).
- Matching and refinement diagnostics: `gatetracker/matching/metrics.py` (`compute_metrics`, `refinement_metrics`, `epipolar_error`, `precision_recall`, …).
- Descriptor loss: `gatetracker/losses/descriptor.py` (`InfoNCELoss`) or triplet margin loss when `DESCRIPTOR_LOSS: triplet`.
- Refinement supervision loss: `gatetracker/losses/geometric.py` (`WeightedSmoothL1Loss`) via `Matcher.compute_refinement_loss`.
- Fusion / contrastive diagnostics: `gatetracker/engine.py` (`compute_architecture_metrics`).
- W&B namespacing and categories: `gatetracker/metrics/logging.py` (`MetricsLogger`, `METRIC_CATEGORIES`).

Default pretrain hyperparameters referenced here come from `config/pretrain.yaml` (e.g. `MAX_EPIPOLAR_DISTANCE: 2`, `FINE_REFINEMENT_LOSS_WEIGHT: 0.25`, `DESCRIPTOR_LOSS: infonce`).

**Math rendering:** display math uses `$$` on its own lines; short expressions use inline `$...$`. Raw less-than and greater-than signs are not used inside inline math (HTML eats them); use `\lt`, `\gt`, or Unicode ≤ / ≥ instead. Avoid placing `$…$` *inside* bold spans. Norms use `\lVert … \rVert` instead of backslash-pipe delimiters for KaTeX. [GitHub](https://github.com) and VS Code (*Markdown: Math*) render these; enable math if you see raw dollar signs.

---

## Naming: canonical name, W&B key, console

**Canonical name**  
The key stored in the per-batch `metrics` dict inside `run_epoch` (before optional epoch-prefixing for CSV aggregates).

**W&B key (batch step)**  
`MetricsLogger` maps flat keys without `/` to the template `{Phase}/{category}/{MetricName}` where `Phase` is `Training`, `Validation`, or `Test`, and `category` is chosen by substring rules in `METRIC_CATEGORIES` (e.g. `Precision` → `matching`). Keys that already contain `/` (e.g. `Training/gate/entropy`) are logged unchanged.

Batch logs also include step scalars: `step/train_batch`, `step/val_batch`, or `step/test_batch` (see `register_wandb_step_axes` in `gatetracker/metrics/logging.py`).

**W&B key (epoch aggregate)**  
After each epoch, means over the epoch’s rows are logged with keys like `Training/epoch/...` / `Validation/epoch/...` plus `step/epoch` (see `engine.py` and `metrics_for_wandb`).

**Console (Rich)**  
Per batch, the logger always prints **`Loss=...`**. Other metrics are appended only for keys **without** `/` in the name, using `abbrev_console_metric_name` in `gatetracker/utils/formatting.py` (initials for CamelCase when the name is longer than six characters). **Architecture keys** (`Training/gate/...`, etc.) are **not** shown in that compact tail; they still appear in W&B and in CSVs.

The **Console tag** column in each section below is the abbreviated label when it differs from the canonical name; otherwise `—` means the full name is used (length ≤ 6 characters) or it is not printed on the console line.

---

## Summary table (pretrain, default config)

| Canonical name | Example W&B key (Training batch) | Console tag |
|-----------------|----------------------------------|---------------|
| `Loss` | `Training/loss/Loss` | `Loss` |
| `DescriptorLoss` | `Training/loss/DescriptorLoss` | `DL` |
| `RefinementLoss` | `Training/loss/RefinementLoss` | `RL` |
| `TrackingLoss` | (usually absent) | `TL` |
| `NRefinementSupervised` | `Training/loss/NRefinementSupervised` | `NRS` |
| `RefinementWeightMean` | `Training/loss/RefinementWeightMean` | `RWM` |
| `InlierCount` | `Training/matching/InlierCount` | `IC` |
| `InlierPercentage` | `Training/matching/InlierPercentage` | `IP` |
| `NTripletsMined` | `Training/matching/NTripletsMined` | `NTM` |
| `Precision` | `Training/matching/Precision` | `Precis` |
| `Recall` | `Training/matching/Recall` | `Recall` |
| `AUCPR` | `Training/matching/AUCPR` | `AUCPR` |
| `EpipolarError` | `Training/matching/EpipolarError` | `EE` |
| `FundamentalError` | `Training/matching/FundamentalError` | `FE` |
| `MDistMean` | `Training/matching/MDistMean` | `MDM` |
| `RefinementActiveFraction` | `Training/refinement/RefinementActiveFraction` | `RAF` |
| `RefinementOffsetMean` | `Training/refinement/RefinementOffsetMean` | `ROM` |
| `RefinementScoreMean` | `Training/refinement/RefinementScoreMean` | `RSM` |
| `CoarseErrorMean` | `Training/refinement/CoarseErrorMean` | `CEM` |
| `RefinedErrorMean` | `Training/refinement/RefinedErrorMean` | `REM` |
| `RefinementGainPx` | `Training/refinement/RefinementGainPx` | `RGP` |
| `RefinementGainRatio` | `Training/refinement/RefinementGainRatio` | `RGR` |
| `RefinementWinRate` | `Training/refinement/RefinementWinRate` | `RWR` |
| `RefinementGainConfidenceCorr` | `Training/refinement/RefinementGainConfidenceCorr` | `RGCC` |
| `Training/optim/LR` | `Training/optim/LR` | `LR` (if ever flat) |
| `Training/optim/GradNorm` | `Training/optim/GradNorm` | `GN` |
| `Training/optim/WeightNorm` | `Training/optim/WeightNorm` | `WN` |
| `gate/entropy` (logged as `Training/gate/entropy`) | `Training/gate/entropy` | — |
| `raw/pos_similarity` | `Training/raw/pos_similarity` | — |
| `fused/margin` | `Training/fused/margin` | — |
| `confidence/mean` | `Training/confidence/mean` | — |

Per-parameter-group learning rates are logged as `Training/optim/LR/{group_name}` (e.g. `hierarchical_fusion`, `fine_feature_head`); console skips `/` keys.

---

## `Loss`

- **W&B (batch):** `Training/loss/Loss` / `Validation/loss/Loss` (training uses the supervised total; validation uses a proxy—see below).
- **Console:** `Loss`.

**Computation**  
Training: scalar value of the tensor backpropagated in the batch:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{desc}} + w_{\text{ref}}\,\mathcal{L}_{\text{ref}} + w_{\text{track}}\,\mathcal{L}_{\text{track}}
$$

with pretrain defaults $w_{\text{track}} \mathcal{L}_{\text{track}} = 0$ (no pairwise tracking). $\mathcal{L}_{\text{desc}}$ is InfoNCE (or triplet margin). $\mathcal{L}_{\text{ref}}$ is the weighted Smooth L1 on refined matches times `FINE_REFINEMENT_LOSS_WEIGHT`, divided by gradient accumulation steps inside the graph.

Validation / test: **`Loss` is set to `EpipolarError`** when not training (engine uses this as the scalar monitored on val for convenience, not the training loss).

**Meaning**  
Primary training objective magnitude (train) or cheap geometric consistency readout (val).

**Optimal trend**  
Train: **decrease** over epochs; unstable spikes or sustained increase suggest LR too high, bad augmentation, or collapsed features. Val: **lower epipolar error is better**; note it is not the same mathematical object as train loss.

**What you learn**  
Whether optimization is stable and whether validation geometry stays consistent with the predicted fundamental model.

---

## `DescriptorLoss`

- **W&B:** `Training/loss/DescriptorLoss` (training only; `None` on val/test in the logged dict).
- **Console:** `DL`.

**Computation (InfoNCE, pretrain default)**  
Masked cross-entropy in cosine-similarity space with temperature $\tau$ (`INFONCE_TEMPERATURE`). For one direction (source → target), with logits $\mathbf{z}_i \in \mathbb{R}^K$, positive index $p_i$, mask entries $M_{ij} \in \lbrace 0,1 \rbrace$, optional weights $w_i$:

$$
\ell_i = -\log \frac{\exp(z_{i,p_i}/\tau)}{\sum_{j \in \mathcal{V}_{i}} \exp(z_{ij}/\tau)}, \qquad
\mathcal{L}_{\text{dir}} = \frac{\sum_i w_i \ell_i}{\sum_i w_i + \epsilon}
$$

with invalid columns masked to $-\infty$ before `logsumexp`. If `INFONCE_SYMMETRIC`, $\mathcal{L}_{\text{desc}} = \frac{1}{2}\bigl(\mathcal{L}_{s \rightarrow t}+\mathcal{L}_{t \rightarrow s}\bigr)$.

**Meaning**  
How well patch embeddings separate positives from negatives in the mined contrastive set.

**Optimal trend**  
**Decrease** toward a floor; very slow descent can mean hard negatives or insufficient capacity. **NaN / explosion** indicates numerical or mining failure.

**What you learn**  
Quality of representation learning and triplet mining difficulty independent of refinement.

---

## `RefinementLoss`

- **W&B:** `Training/loss/RefinementLoss`.
- **Console:** `RL`.

**Computation**  
`WeightedSmoothL1Loss` between refined predicted target pixels and pseudo-GT targets, averaged over matches with optional photometric confidence weights and validity mask, then multiplied by `FINE_REFINEMENT_LOSS_WEIGHT` and divided by gradient accumulation steps in the graph.

Per spatial coordinate, Smooth L1 with parameter $\beta$:

$$
\text{SL1}(d) =
\begin{cases}
\frac{1}{2}\, d^2 / \beta & \text{if } \lvert d \rvert \leq \beta \\
\lvert d \rvert - \frac{1}{2}\beta & \text{if } \lvert d \rvert \gt \beta
\end{cases}
$$

with $d$ the component error; summed over the two coordinates per match, then confidence-weighted mean.

**Meaning**  
How much the fine head must move predictions to match geometry-supervised targets.

**Optimal trend**  
**Decrease** early; a **very small** value with **stagnant matching metrics** can mean the refinement head is ignored or supervision is too easy.

**What you learn**  
Whether sub-pixel refinement is learning useful corrections vs. overfitting noise.

---

## `TrackingLoss`

- **W&B:** `Training/loss/TrackingLoss` when pairwise tracking runs.
- **Console:** `TL`.

**Computation**  
Only if `pairwise_tracking_enabled(config)`; otherwise absent. Weighted sum of cycle, descriptor similarity, and coarse-to-fine regularization terms from `compute_pairwise_tracking_losses` (`gatetracker/tracking/losses.py`), logged as detached scalar.

**Pretrain default:** typically **not logged** (phase is `pretrain` and `TRACKING_MODE` is false).

**Meaning**  
Self-supervised temporal / cycle consistency for the optional tracking head.

**Optimal trend**  
If enabled: **decrease**; large values with frozen descriptors may indicate incompatible window or weights.

**What you learn**  
Whether the optional pairwise tracker is aligned with descriptors; **not part of default pretrain**.

---

## `NRefinementSupervised`

- **W&B:** `Training/loss/NRefinementSupervised`.
- **Console:** `NRS`.

**Computation**  
Integer count of matches with finite predictions and targets passing validity / weight masks in `compute_refinement_loss`.

**Meaning**  
How many correspondences actually contribute refinement supervision each batch.

**Optimal trend**  
**Stable, sufficiently large** relative to batch; **near zero** means refinement is starved (masks, confidence, or method gating).

**What you learn**  
Data and gating sufficiency for the refinement branch.

---

## `RefinementWeightMean`

- **W&B:** `Training/loss/RefinementWeightMean`.
- **Console:** `RWM`.

**Computation**  
Mean supervision weight (or `1.0` if weights absent) over valid refinement matches.

**Meaning**  
Average confidence placed on refinement targets.

**Optimal trend**  
Context-dependent; **collapse toward 0** with `USE_CORRESPONDENCE_CONFIDENCE` suggests bad confidence calibration.

**What you learn**  
Whether photometric / confidence weighting is down-weighting most matches.

---

## `InlierCount`

- **W&B:** `{Phase}/matching/InlierCount`.
- **Console:** `IC`.

**Computation**  
`inliers.count_nonzero().item()` after `Matcher.RANSAC` on the current correspondence set: count of inlier flags in the robust model fit.

**Meaning**  
How many putative matches survive the geometric consensus.

**Optimal trend**  
**Higher** usually better **if** quality metrics (precision, epipolar error) stay good; **high count with poor precision** indicates permissive RANSAC.

**What you learn**  
Robustness of the putative set to epipolar geometry; pipeline health for fundamental estimation.

---

## `InlierPercentage`

- **W&B:** `{Phase}/matching/InlierPercentage`.
- **Console:** `IP`.

**Computation**  
`InlierCount` / $N_{\mathrm{matches}}$, where $N_{\mathrm{matches}}$ is the number of elements in the inlier tensor.

**Meaning**  
Fraction of matches consistent with the estimated fundamental matrix.

**Optimal trend**  
**Higher** is generally better for stable geometry, subject to the same caveat as `InlierCount`.

**What you learn**  
Signal-to-noise in correspondence scoring and RANSAC thresholding.

---

## `NTripletsMined`

- **W&B:** `Training/matching/NTripletsMined` (training only).
- **Console:** `NTM`.

**Computation**  
`descriptor_pairs_mined / batch_size`, where `descriptor_pairs_mined` is the number of contrastive pairs (InfoNCE positives) or anchors (triplet path) mined in the batch.

**Meaning**  
Average mined structured pairs per sample—mining throughput.

**Optimal trend**  
**Stable** in line with `TRIPLETS_TO_MINE` and batch size; **near zero** means mining failed (empty triplets, masking, or score thresholds too strict).

**What you learn**  
Whether contrastive supervision is actually populated each step.

---

## `Precision`

- **W&B:** `{Phase}/matching/Precision`.
- **Console:** `Precis`.

**Computation** (`precision_recall` with `reduction="mean"`)  
For each match $i$, geometric distance $d_i$: Euclidean to GT target if available, else per-point epipolar error using `fundamental_for_epipolar` (GT $F$ if valid batch-wide, else predicted $F$). Binary label $y_i \in \lbrace 0,1 \rbrace$ with $y_i = 1$ when $d_i \lt D_{\max}$, where $D_{\max}$ is the config value `MAX_EPIPOLAR_DISTANCE` (pixels). Sort matches by descending confidence; cumulative true positives $\mathrm{TP}_k$; precision curve $P_k = \mathrm{TP}_k / k$. The reported value is the **mean** of $P_k$ over all ranked positions (not precision at a single operating point).

**Meaning**  
Ranking quality of confidences against a geometric correctness threshold.

**Optimal trend**  
**Increase** toward 1; **flat near random** with high AUCPR still possible—interpret alongside AUCPR.

**What you learn**  
Whether high scores align with geometrically plausible matches under your distance definition.

---

## `Recall`

- **W&B:** `{Phase}/matching/Recall`.
- **Console:** `Recall`.

**Computation**  
Same curve as precision, with recall $R_k = \mathrm{TP}_k / (\sum_i y_i + \epsilon)$; the reported value is the **mean** of $R_k$ over ranks.

**Meaning**  
How completely confident predictions cover true positives as you sweep the ranking.

**Optimal trend**  
**Increase** toward 1.

**What you learn**  
Coverage of correct matches in the confident tail; useful with Precision for imbalance.

---

## `AUCPR`

- **W&B:** `{Phase}/matching/AUCPR`.
- **Console:** `AUCPR`.

**Computation**  
Trapezoidal integral $\int R\, dP$ (implemented as `torch.trapz(precision_curve, recall_curve)`) along the same sorted-by-confidence curve.

**Meaning**  
Scalar summary of precision–recall tradeoff for geometric correctness vs. score.

**Optimal trend**  
**Increase** toward 1.

**What you learn**  
Overall matching quality independent of a single threshold; good for comparing runs.

---

## `EpipolarError`

- **W&B:** `{Phase}/matching/EpipolarError`.
- **Console:** `EE`.

**Computation**  
For homogeneous points $\tilde{\mathbf{x}}_1,\tilde{\mathbf{x}}_2$ and per-match fundamental $\mathbf{F}$, epipolar line on image 2: $\mathbf{l}_2 = \mathbf{F}\tilde{\mathbf{x}}_1$; symmetric point-to-line distances averaged:

$$
e = \frac{1}{2}\left(
\frac{\lvert \mathbf{l}_2^\top \tilde{\mathbf{x}}_2 \rvert}{\sqrt{a_2^2+b_2^2+\delta}} +
\frac{\lvert \mathbf{l}_1^\top \tilde{\mathbf{x}}_1 \rvert}{\sqrt{a_1^2+b_1^2+\delta}}
\right)
$$

with $\mathbf{l}_1 = \mathbf{F}^\top \tilde{\mathbf{x}}_2$, line coefficients $(a_j,b_j,c_j)$ from $\mathbf{l}_j$, small $\delta$ for numerical stability in code (`1e-8`). Mean over all matches uses **predicted** `fundamental_pred`.

**Meaning**  
Average symmetric epipolar distance in **pixels** (given image scaling) for the current matches and estimated $F$.

**Optimal trend**  
**Decrease** toward 0; compare to `MAX_EPIPOLAR_DISTANCE` (pixels) used for inlier labeling.

**What you learn**  
Consistency of correspondences with the recovered epipolar geometry; primary **validation proxy for `Loss`**.

---

## `FundamentalError`

- **W&B:** `{Phase}/matching/FundamentalError` when not `None`.
- **Console:** `FE`.

**Computation**  
If batch GT fundamentals are valid (non-degenerate Frobenius norm), normalized matrices $\hat{\mathbf{F}}_{\text{pred}}$ and $\hat{\mathbf{F}}_{\text{gt}}$ are compared with sign ambiguity:

$$
e_F = \min\left(
\lVert \hat{\mathbf{F}}_{\text{pred}}-\hat{\mathbf{F}}_{\text{gt}}\rVert_F,\ 
\lVert \hat{\mathbf{F}}_{\text{pred}}+\hat{\mathbf{F}}_{\text{gt}}\rVert_F
\right)
$$

optionally scaled (percentage mode in `fundamental_error`). If no valid GT, metric is **`None`** and omitted from logs.

**Meaning**  
How close predicted $F$ is to dataset ground truth when available.

**Optimal trend**  
**Decrease**; **absent** on datasets without GT poses is expected.

**What you learn**  
Explicit pose / calibration supervision signal vs. self-supervised geometry.

---

## `MDistMean`

- **W&B:** `{Phase}/matching/MDistMean` when not `None`.
- **Console:** `MDM`.

**Computation**  
Mean Euclidean norm $\lVert \mathbf{p}_{\text{pred}}-\mathbf{p}_{\text{gt}}\rVert_2$ over matches when `true_pixels_matched` exists (after synthetic GT alignment in training).

**Meaning**  
Average pixel error vs. pseudo- or true correspondences.

**Optimal trend**  
**Decrease**; **`None`** when no GT targets (e.g. some eval paths).

**What you learn**  
Direct localization quality of matches in image space.

---

## `RefinementActiveFraction`

- **W&B:** `{Phase}/refinement/RefinementActiveFraction`.
- **Console:** `RAF`.

**Computation**  
Mean of boolean `active_mask` over refined matches from `latest_refinement_state`.

**Meaning**  
Fraction of matches for which refinement is considered active.

**Optimal trend**  
Context-dependent; **collapse to 0** or **1** fixed for long spans can indicate gating bugs or disabled refinement.

**What you learn**  
How aggressively the pipeline uses the refinement head.

---

## `RefinementOffsetMean`

- **W&B:** `{Phase}/refinement/RefinementOffsetMean`.
- **Console:** `ROM`.

**Computation**  
Mean $\lVert \mathbf{p}_{\text{ref}}-\mathbf{p}_{\text{coarse}}\rVert_2$ over matches.

**Meaning**  
Average magnitude of the refinement displacement vector.

**Optimal trend**  
**Moderate**: near-zero forever may mean refinement is unused; **very large** vs. patch scale may mean instability.

**What you learn**  
How far the fine head moves coarse matches on average.

---

## `RefinementScoreMean`

- **W&B:** `{Phase}/refinement/RefinementScoreMean`.
- **Console:** `RSM`.

**Computation**  
Mean of `refined_scores` stored with refinement state.

**Meaning**  
Average refined confidence or quality score (implementation-specific to the matcher).

**Optimal trend**  
Interpret relative to other runs; **NaN** if empty state.

**What you learn**  
Whether refinement increases match confidence on average.

---

## `CoarseErrorMean`

- **W&B:** `{Phase}/refinement/CoarseErrorMean`.
- **Console:** `CEM`.

**Computation**  
Mean $\lVert \mathbf{p}_{\text{coarse}}-\mathbf{p}_{\text{gt}}\rVert_2$ when GT targets exist.

**Meaning**  
Baseline localization error before refinement.

**Optimal trend**  
**Decrease** over training (coarse matching improving).

**What you learn**  
Quality of the coarse correspondence field independent of refinement.

---

## `RefinedErrorMean`

- **W&B:** `{Phase}/refinement/RefinedErrorMean`.
- **Console:** `REM`.

**Computation**  
Mean $\lVert \mathbf{p}_{\text{ref}}-\mathbf{p}_{\text{gt}}\rVert_2$.

**Meaning**  
Localization error after refinement.

**Optimal trend**  
**Decrease**; should be **≤** coarse error on average when refinement helps.

**What you learn**  
Net gain from the fine head vs. coarse alone.

---

## `RefinementGainPx`

- **W&B:** `{Phase}/refinement/RefinementGainPx`.
- **Console:** `RGP`.

**Computation**  
Mean of $(e_{\text{coarse}} - e_{\text{refined}})$ in pixels, with $e$ the per-match L2 error.

**Meaning**  
Average pixel improvement from refinement.

**Optimal trend**  
**Positive** and stable when refinement is beneficial; **negative** means refinement hurts on average.

**What you learn**  
Direct net effect of refinement on supervised pixel error.

---

## `RefinementGainRatio`

- **W&B:** `{Phase}/refinement/RefinementGainRatio`.
- **Console:** `RGR`.

**Computation**  
Same batch as logged scalars, implemented as mean per-match gain over mean coarse error with a small denominator floor:

$$
\mathrm{RGR} = \frac{\mathbb{E}[e_{\text{coarse}} - e_{\text{refined}}]}{\mathbb{E}[e_{\text{coarse}}] + \epsilon},
\quad \epsilon = 10^{-6}.
$$

**Meaning**  
Fractional reduction of error relative to coarse baseline.

**Optimal trend**  
**Positive**; values near **0** mean little relative gain.

**What you learn**  
Whether gains are meaningful relative to remaining coarse error.

---

## `RefinementWinRate`

- **W&B:** `{Phase}/refinement/RefinementWinRate`.
- **Console:** `RWR`.

**Computation**  
Fraction of matches with $e_{\text{coarse}} \gt e_{\text{refined}}$ (strict improvement).

**Meaning**  
Per-match consistency of refinement helping.

**Optimal trend**  
**Increase** toward 1; **near 0.5** behaves like random sign.

**What you learn**  
Robustness of refinement: whether it helps most matches or only a subset.

---

## `RefinementGainConfidenceCorr`

- **W&B:** `{Phase}/refinement/RefinementGainConfidenceCorr`.
- **Console:** `RGCC`.

**Computation**  
Pearson correlation coefficient between centered `refined_scores` and centered per-match gain $(e_{\text{coarse}}-e_{\text{refined}})$, when both have non-zero variance.

**Meaning**  
Whether higher refined confidence aligns with larger pixel improvements.

**Optimal trend**  
**Positive** correlation is desirable if scores are meant to rank good refinements; **undefined** when variance is too small.

**What you learn**  
Calibration of refinement confidence vs. actual utility.

---

## `Training/optim/LR` (and per-group `Training/optim/LR/{group}`)

- **W&B:** `Training/optim/LR`, `Training/optim/LR/hierarchical_fusion`, `Training/optim/LR/fine_feature_head`, etc.
- **Console:** not printed (keys contain `/`).

**Computation**  
Scalar learning rate from `optimizer.param_groups[0]['lr']` and each group’s `lr` after schedulers / warmup (`engine.py`, `_optimizer_lr_metrics`).

**Meaning**  
Effective optimization step sizes per module family.

**Optimal trend**  
Follows schedule (cosine decay, etc.); **unexpected jumps** indicate scheduler or warmup bugs.

**What you learn**  
Whether fusion vs. fine head vs. backbone receive intended relative rates (`LR_FUSION`, `LR_FINE_FEATURE`, …).

---

## `Training/optim/GradNorm`

- **W&B:** `Training/optim/GradNorm`.
- **Console:** `GN` (only for flat historical naming; current key has `/` so **omitted from console tail**—the metric is still logged).

**Computation**  
After `loss.backward()`, `get_norms` walks every parameter tensor and sums all absolute entries (total L1 mass of weights, not of gradients). Despite the name “GradNorm”, `gatetracker/utils/optimization.py::get_norms` implements that sum-of-$\lvert \cdot \rvert$ statistic plus an L2 norm of the full parameter vector for the second return value.

**Correction / caveat**  
The logged “GradNorm” is **not** a gradient norm such as $\lVert \nabla \mathcal{L} \rVert_2$. Treat it as a coarse scale signal unless renamed in code.

**Optimal trend**  
Stable order of magnitude; **exploding** values with NaN losses indicate instability.

**What you learn**  
Rough parameter scale dynamics; for true gradient norms you would log `torch.nn.utils.clip_grad_norm_` output separately.

---

## `Training/optim/WeightNorm`

- **W&B:** `Training/optim/WeightNorm`.
- **Console:** `WN` (same `/` caveat as GradNorm).

**Computation**  
Euclidean norm of all parameters viewed as one vector: $\sqrt{\sum_{p} \lVert p \rVert_2^2}$.

**Meaning**  
Global weight magnitude.

**Optimal trend**  
**Stable** or slowly changing; **runaway growth** may indicate missing weight decay or numerical blow-up.

**What you learn**  
Regularization / capacity usage across the full model.

---

## `gate/entropy` (W&B: `Training/gate/entropy`, …)

- **W&B:** `{Phase}/gate/entropy`.
- **Console:** — (not in compact line).

**Computation**  
From hierarchical fusion layer weights $w_{\ell}$ (non-negative, normalized per location): Shannon entropy $H = -\sum_\ell w_\ell \log w_\ell$, averaged over locations and batches, then **divided** by $\log L$ so the value is roughly scaled between 0 and 1.

**Meaning**  
How “spread” the fusion gate is across DINO layers (vs. peaked on one layer).

**Optimal trend**  
**Neither extreme forever**: all mass on one layer → low entropy; uniform mixing → high entropy. Sudden **collapse** after being spread can mean a routing bug or overfitting one scale.

**What you learn**  
Whether the model uses the intended multi-scale representation.

---

## `gate/max_weight`

- **W&B:** `{Phase}/gate/max_weight`.
- **Console:** —.

**Computation**  
Mean over matches of $\max_\ell w_\ell$ at each location.

**Meaning**  
How dominant the strongest layer is.

**Optimal trend**  
Extremes near **1** always imply single-layer routing.

**What you learn**  
Peaked vs. distributed fusion behavior.

---

## `gate/weight_shallow` and `gate/weight_deep`

- **W&B:** `{Phase}/gate/weight_shallow`, `{Phase}/gate/weight_deep`.
- **Console:** —.

**Computation**  
Mean weights in the first and second halves of layer indices (split at `num_layers // 2`).

**Meaning**  
Relative preference for shallow vs. deep ViT features in the fusion.

**Optimal trend**  
Dataset-dependent; **watch for sudden flips** when changing backbone or resolution.

**What you learn**  
Inductive bias of the fusion toward texture vs. semantics at patch scale.

---

## `gate/effective_layer`

- **W&B:** `{Phase}/gate/effective_layer`.
- **Console:** —.

**Computation**  
For each location, effective layer index $\tilde{\ell} = \sum_\ell w_\ell \cdot \ell$; metric is mean over locations of **std across spatial dimension** of $\tilde{\ell}$ (see `compute_architecture_metrics`).

**Meaning**  
Spatial variability of which layer the gate emphasizes.

**Optimal trend**  
**Too low** may mean spatially uniform fusion; **very high** may mean noisy routing.

**What you learn**  
Whether the gate spatially adapts layer mixing for endoscopic structure.

---

## `raw/pos_similarity`

- **W&B:** `{Phase}/raw/pos_similarity`.
- **Console:** —.

**Computation**  
Mean cosine similarity between **raw** (pre-fusion) source anchor and target positive patch embeddings over mined triplets.

**Meaning**  
Baseline appearance agreement before hierarchical fusion.

**Optimal trend**  
**Increase** toward 1 during training (easier positives).

**What you learn**  
How much signal exists in the frozen / pre-fusion representation for matching.

---

## `raw/margin`

- **W&B:** `{Phase}/raw/margin`.
- **Console:** —.

**Computation**  
Mean of $(\cos_{\text{pos}}^{\text{raw}} - \cos_{\text{neg}}^{\text{raw}})$ over triplets.

**Meaning**  
Raw embedding separability between positive and negative patches.

**Optimal trend**  
**Increase**; **negative** means negatives are closer than positives on average (bad).

**What you learn**  
Triplet hardness and raw descriptor quality.

---

## `fused/pos_similarity`

- **W&B:** `{Phase}/fused/pos_similarity`.
- **Console:** —.

**Computation**  
Same as raw, but on **fused** anchor/positive embeddings.

**Meaning**  
Agreement after fusion.

**Optimal trend**  
**Increase**; should generally track or exceed raw if fusion helps.

**What you learn**  
Whether fusion improves positive agreement for mined pairs.

---

## `fused/margin`

- **W&B:** `{Phase}/fused/margin`.
- **Console:** —.

**Computation**  
Mean $(\cos_{\text{pos}}^{\text{fused}} - \cos_{\text{neg}}^{\text{fused}})$.

**Meaning**  
Fused separability for contrastive learning.

**Optimal trend**  
**Increase**; compare to `raw/margin` for **gain**.

**What you learn**  
Direct evidence that fusion aids discrimination.

---

## `fused/gain`

- **W&B:** `{Phase}/fused/gain`.
- **Console:** —.

**Computation**  
Difference of batch means: mean fused cosine margin minus mean raw cosine margin over mined triplets (same construction as in `compute_architecture_metrics`, scalar `fused/gain`).

**Meaning**  
Average improvement in margin from fusion vs. raw embeddings.

**Optimal trend**  
**Positive** sustained values support the fusion design; **negative** sustained values suggest fusion hurts or is misconfigured.

**What you learn**  
Net contribution of hierarchical fusion to the training objective geometry.

---

## `confidence/mean`, `confidence/std`, `confidence/high_low_diff`

- **W&B:** `{Phase}/confidence/mean`, etc.
- **Console:** —.

**Computation**  
Statistics on `triplet_confidence` when present: mean, standard deviation, and difference of mean fused margins between high- vs. low-confidence halves split at the median confidence.

**Meaning**  
Distribution of learned correspondence confidence and whether confident triplets have larger fused margins.

**Optimal trend**  
`high_low_diff` **positive** suggests confidence is informative; **near zero** means confidence is uncalibrated vs. margin.

**What you learn**  
Whether confidence gating for InfoNCE weighting (`contrastive_weights`) is aligned with hard/easy triplets.

---

## CSV-only bookkeeping (not primary interpretive metrics)

- **`Step/batch`, `Step/valbatch`, `Step/epoch`, `Step/idx`:** written into DataFrames for aligning rows when exporting `training_metrics.csv` / `validation_metrics.csv`. They are **stripped from W&B batch payloads** (see `MetricsLogger.log_batch`) in favor of `step/train_batch`, `step/val_batch`, `step/epoch`.

---

## Optional extras (not exhaustive)

- **`wandb.watch`:** may log gradient / parameter histograms under separate keys (`Gradients/*` patterns). Interpret per W&B docs.
- **Images:** `Training/images/*` etc. are media, not scalars; they visualize matches, embeddings, and geometry for qualitative diagnosis.

---

## Quick read on a pretrain run

1. **Train:** `Training/loss/Loss`, `DescriptorLoss`, `RefinementLoss` ↓; `AUCPR` ↑; `EpipolarError` ↓.  
2. **Fusion sanity:** `fused/gain` > 0 on average; `gate/entropy` not collapsed unexpectedly.  
3. **Refinement:** `RefinementGainPx` > 0, `RefinedErrorMean` ≤ `CoarseErrorMean`.  
4. **Val:** `Validation/matching/EpipolarError` and `Validation/loss/Loss` (same scalar) ↓ for plateau scheduler / early stopping.  
5. **Mining:** `NTripletsMined` stable; `NRefinementSupervised` not near zero.

If any metric name or formula drifts from code, prefer the implementation in the files cited at the top of this document.

---

## Phase: Tracking

This subsection covers **`PHASE: tracking`**: training the **TemporalRefinementNetwork** while the **Matcher / descriptor backbone stays frozen** (`TemporalTracker` in `gatetracker/tracking/tracker.py`). Batches are short frame windows from `SequenceWindowDataset`; optimization uses `tracking_trainloop` → `_run_tracking_epoch` in `gatetracker/engine.py`.

**Sources of truth**

- Per-batch loss and metrics: `TemporalTracker.training_step` → `compute_temporal_tracking_losses`, optional `temporal_supervised_losses` in `gatetracker/tracking/losses.py`.
- Loop, optimizer, epoch scalars: `gatetracker/engine.py` (`_run_tracking_epoch`).
- Optional **StereoMIS** TAP-Vid-style numbers + videos: `_tracking_validate_stereomis_gt` → `compute_tap_metrics` in `gatetracker/tracking/metrics.py`.
- Defaults and toggles: `config/tracking.yaml` (pseudo-GT block, `TRACKING_STEREOMIS_VAL`, …).

**Math:** display math uses `$$` on its own lines; inline math uses `$...$`. Same KaTeX caveats as in the pretrain section.

---

### Supervision pipeline (what is optimized)

**Stage 1 — frozen coarse association.** For each time index $t$, query descriptors are sampled on the **patch grid** at frame $0$ and matched globally to frame $t$ using inner products over patch embeddings (`_global_coarse_match`). This yields **coarse tracks** $\mathbf{p}^{\mathrm{coarse}}_{b,q,t} \in \mathbb{R}^2$ with batch $b$, query $q$, time $t$ (shapes: `frames` $[B,T,3,H,W]$, coarse tracks $[B,Q,T,2]$).

**Stage 2 — trainable temporal refinement.** The **TemporalRefinementNetwork** refines coarse tracks using local correlation over **fine feature maps** (requires `REFINEMENT_METHOD: feature_softargmax` so `fine_feature_maps` exist). Outputs:

- **Forward tracks** $\hat{\mathbf{p}}_{b,q,t}$ (`tracks_fwd`, $[B,Q,T,2]$).
- **Visibility logits** $\ell_{b,q,t}$ (`vis_fwd`, $[B,Q,T]$).

**Self-supervised objective (real windows or synthetic windows).** Let $\mathcal{L}_{\mathrm{cycle}}$, $\mathcal{L}_{\mathrm{smooth}}$, $\mathcal{L}_{\mathrm{desc}}$, $\mathcal{L}_{\mathrm{feat}}$, $\mathcal{L}_{\mathrm{vis}}$ be the terms returned by `compute_temporal_tracking_losses` (unweighted components before the final sum). With config weights $w_{\mathrm{cycle}}, w_{\mathrm{smooth}}, w_{\mathrm{desc}}, w_{\mathrm{feat}}, w_{\mathrm{vis}}$ (`TEMPORAL_*_LOSS_WEIGHT` / `TEMPORAL_VIS_REG_WEIGHT`, defaults in code: $1.0, 0.3, 0.5, 0.5, 0.1$) and an optional scale $\alpha =$ `PSEUDO_GT_SYNTH_SELF_SUP_SCALE` on the **core** self-supervised sum:

$$
\mathcal{L}_{\mathrm{self}} = \alpha \bigl( w_{\mathrm{cycle}} \mathcal{L}_{\mathrm{cycle}} + w_{\mathrm{smooth}} \mathcal{L}_{\mathrm{smooth}} + w_{\mathrm{desc}} \mathcal{L}_{\mathrm{desc}} + w_{\mathrm{feat}} \mathcal{L}_{\mathrm{feat}} \bigr) + w_{\mathrm{vis}} \mathcal{L}_{\mathrm{vis}}.
$$

- **Cycle:** forward track vs. **backward** track through time (endpoint round-trip to queries at $t{=}0$ plus intermediate agreement), weighted by $\sigma(\ell_{:, :, 0})$ and optional mask (`temporal_cycle_consistency_loss`).
- **Smoothness:** mean $\lVert \hat{\mathbf{p}}_{t+1} - 2\hat{\mathbf{p}}_t + \hat{\mathbf{p}}_{t-1}\rVert_2$ (second difference), weighted by visibility across three frames (`temporal_smoothness_loss`).
- **Descriptor consistency:** mean $1 - \cos\bigl(\mathbf{e}^{\mathrm{query}}_{b,q}, \mathbf{e}^{\mathrm{target}}_{b,q,t}\bigr)$ over $t$ with frozen descriptors (`temporal_descriptor_consistency_loss`).
- **Feature persistence:** $\lVert \mathbf{f}^{\mathrm{query}}_{b,q} - \mathbf{f}^{\mathrm{tracked}}_{b,q,t}\rVert_2$ on fine features (`feature_persistence_loss`).
- **Visibility regularization:** temporal smoothness of $\sigma(\ell)$ plus negative entropy to reduce all-visible collapse (`visibility_regularization_loss`).

**Pseudo–ground-truth hybrid (optional).** If `PSEUDO_GT_SUP_LAMBDA_MAX` $\gt 0$, `PSEUDO_GT_MIX` $\gt 0$, `geometry_pipeline` is set, and a Bernoulli trial succeeds, frame $0$ of the batch is used to predict depth + intrinsics (`geometry_pipeline.compute_geometry`), and `PseudoGTGenerator` builds a **synthetic multi-frame stack** with known tracks and visibility. Query pixels come from the generator grid (subsampled to `TRACKING_NUM_QUERY_POINTS`). For those batches:

- **Cycle loss weight is zeroed** (`cycle_weight_scale = 0`); no backward refinement pass is run (`tracks_bwd = tracks_fwd.detach()`), so $w_{\mathrm{cycle}}\mathcal{L}_{\mathrm{cycle}}$ does not contribute to $\mathcal{L}_{\mathrm{self}}$ (the logged `loss_cycle` scalar may still be non-zero as a diagnostic).
- A **composite supervision mask** $M_{b,q,t}$ combines pseudo visibility, in-bounds checks, and RGB holemask samples (`composite_supervision_mask`, `validity_at_tracks_bqt`).
- **Supervised loss** (`temporal_supervised_losses`): masked Smooth L1 on $\lVert \hat{\mathbf{p}} - \mathbf{p}^{\mathrm{gt}}\rVert_2$ (Huber $\beta =$ `TEMPORAL_SUP_HUBER_BETA`) plus BCE-with-logits on visibility vs. `vis_target` (optionally appearance-aware).

**Curriculum on $\lambda$.** Let $\lambda_{\max} =$ `PSEUDO_GT_SUP_LAMBDA_MAX` and $E_{\mathrm{cur}} =$ `PSEUDO_GT_CURRICULUM_EPOCHS`. For a 0-based epoch index $e$:

$$
\lambda(e) = \lambda_{\max} \cdot \min\!\left(1,\ \frac{e+1}{E_{\mathrm{cur}}}\right).
$$

The **scalar optimized** on pseudo batches is the convex blend

$$
\mathcal{L}_{\mathrm{total}} = (1 - \lambda(e))\,\mathcal{L}_{\mathrm{self}} + \lambda(e)\,\mathcal{L}_{\mathrm{sup}},
$$

with $\mathcal{L}_{\mathrm{sup}} = w_{\mathrm{pos}}\mathcal{L}_{\mathrm{sup,pos}} + w_{\mathrm{vis}}\mathcal{L}_{\mathrm{sup,vis}}$ (`TEMPORAL_SUP_POS_WEIGHT`, `TEMPORAL_SUP_VIS_WEIGHT`). When the pseudo branch is inactive, $\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{self}}$ and $\lambda$ is logged as $0$.

---

### Training vs. validation (same code path, different side effects)

| Aspect | Training | Validation |
|--------|-----------|--------------|
| `refinement_net` mode | `train()` | `eval()` |
| Autograd | `loss_total.backward()`, Adam step on refinement net only; grad clip $1.0$ | `torch.no_grad()` — **no** weight updates |
| Data order | `shuffle=True` on the sequence-window loader | `shuffle=False` |
| `training_step` | Same forward + loss (including random pseudo-GT Bernoulli and dropout behavior in train mode) | Same call graph; pseudo-GT can still trigger stochastically if enabled |
| Extra metrics | — | After the epoch: **`_tracking_validate_stereomis_gt`** — TAP-Vid-style $\delta_{\mathrm{avg}}$, OA, AJ on a StereoMIS sequence, plus optional W&B **videos** (`grid_video`, `gt_vs_pred_video`) under `Validation/stereomis_gt/...` |
| LR / early stop | — | Mean batch `Loss` over the val epoch drives `LRschedulerPlateau` (if configured) and **early stopping** on the refinement net |

**Interpretation:** logged **batch** metrics on validation are still **self-supervised (+ optional pseudo)** losses on held-out windows, not “pure” supervised tracking error. **Supervised-style** numbers appear under **`Validation/stereomis_gt/*`** when that block runs and GT exists.

---

### Naming: tracking batch keys, W&B, console

**Batch dict (into `MetricsLogger.log_batch`)**

- Prefixed metrics: `{Phase}/tracking/{name}` (e.g. `Training/tracking/loss_cycle`). These already contain `/`, so they are logged to W&B **unchanged** (see `_namespace_metrics` in `gatetracker/metrics/logging.py`).
- **`Loss`:** scalar backpropagated total $\mathcal{L}_{\mathrm{total}}$ (Python float). Console always prints `Loss=...` first.
- **`Step/epoch` or `Step/val_epoch`:** CSV alignment; stripped from W&B batch payloads like other `Step/*` keys.

**W&B batch step axis:** `step/train_batch` / `step/val_batch` (registered against `{Phase}/tracking/*` via `register_wandb_step_axes`).

**Console (Rich):** After `Loss`, any key **without** `/` would be abbreviated; tracking uses only prefixed keys **plus** `Loss`. Substrings after `{Phase}/tracking/` are passed to `abbrev_console_metric_name` (snake_case → initials of each segment, max length 6). Examples: `loss_cycle` → `lc`, `sup_mask_fraction` → `smf`, `pseudo_gt_active` → `pga`.

**Epoch scalar:** `{Phase}/tracking/epoch_loss` — mean of per-batch `Loss` over the completed epoch (both train and val epochs log it when at least one batch ran).

---

### Summary table (tracking — per-batch `training_step` metrics)

| Canonical name | Example W&B key (Training batch) | Console tag |
|----------------|----------------------------------|---------------|
| `Loss` | (flat; not `Phase/category/Loss` — logger keeps `Loss` as-is in wandb dict after namespace pass-through for slash keys; **note:** `Loss` has no `/` so becomes `Training/loss/Loss` — see below) | `Loss` |
| `loss_cycle` | `Training/tracking/loss_cycle` | `lc` |
| `loss_smooth` | `Training/tracking/loss_smooth` | `ls` |
| `loss_desc` | `Training/tracking/loss_desc` | `ld` |
| `loss_feat` | `Training/tracking/loss_feat` | `lf` |
| `loss_vis` | `Training/tracking/loss_vis` | `lv` |
| `loss_total` | `Training/tracking/loss_total` | `lt` |
| `pseudo_gt_active` | `Training/tracking/pseudo_gt_active` | `pga` |
| `pseudo_lambda` | `Training/tracking/pseudo_lambda` | `pl` |
| `loss_sup_pos` | `Training/tracking/loss_sup_pos` | `lsp` |
| `loss_sup_vis` | `Training/tracking/loss_sup_vis` | `lsv` |
| `loss_sup_total` | `Training/tracking/loss_sup_total` | `lst` |
| `sup_mask_fraction` | `Training/tracking/sup_mask_fraction` | `smf` |

**Note on `Loss` vs. `loss_total`:** both reflect the same blended objective after the pseudo block; `loss_total` duplicates the detached scalar inside the `metrics` dict for convenience. In `MetricsLogger._namespace_metrics`, the flat key `Loss` is categorized and becomes **`Training/loss/Loss`** (or `Validation/loss/Loss`) on W&B — so the primary train chart for the optimization scalar is **`{Phase}/loss/Loss`**, while the decomposition lives under **`{Phase}/tracking/*`**.

**Epoch aggregate (not in batch table):** `{Phase}/tracking/epoch_loss`.

**StereoMIS validation (end of each val epoch, if enabled and data resolve):** `Validation/stereomis_gt/{sequence}/delta_avg`, `.../OA`, `.../AJ`, optional `Validation/stereomis_gt/grid_video`, `Validation/stereomis_gt/gt_vs_pred_video` (media objects), all logged with `step/epoch` in the same payload (`_wandb_epoch_axis_dict`).

---

### `Loss` (tracking)

- **W&B (batch):** `{Phase}/loss/Loss` (from namespacing the flat `Loss` key).
- **Console:** `Loss`.

**Computation**  
Same scalar tensor as `loss_dict["loss_total"]` after optional pseudo-GT blending: $\mathcal{L}_{\mathrm{total}}$ above.

**Meaning**  
The single objective minimized on training batches (self-supervised mix, plus curriculum-weighted supervised term on pseudo batches).

**Optimal trend**  
**Decrease** on average across epochs for train; validation batch mean **decrease** is used for plateau / early stopping but is **not** the same as StereoMIS $\delta_{\mathrm{avg}}$.

**What you learn**  
Whether temporal refinement is learning stable trajectories under the chosen mix of self-supervision and pseudo-GT.

---

### `loss_cycle`

- **W&B:** `{Phase}/tracking/loss_cycle`.
- **Console:** `lc`.

**Computation**  
Unweighted mean cycle + waypoint error (`temporal_cycle_consistency_loss`) with optional visibility and `self_sup_mask` reweighting; **multiplied by** `TEMPORAL_CYCLE_LOSS_WEIGHT` and `cycle_weight_scale` **inside** the weighted sum (the logged value is the **unweighted** component before $w_{\mathrm{cycle}}$).

**Meaning**  
Temporal consistency of forward vs. backward trajectories when the backward pass is active.

**Optimal trend**  
**Decrease** when $w_{\mathrm{cycle}} > 0$; on pseudo batches $w_{\mathrm{cycle}}=0$ so the term is irrelevant to gradients.

---

### `loss_smooth`

- **W&B:** `{Phase}/tracking/loss_smooth`.
- **Console:** `ls`.

**Computation**  
Mean norm of discrete acceleration of tracks ($[B,Q,T,2]$), with visibility / mask weights (`temporal_smoothness_loss`).

**Meaning**  
Penalizes jittery tracks.

**Optimal trend**  
**Decrease**, balanced against cycle/descriptor terms so tracks do not become overly inertial.

---

### `loss_desc`

- **W&B:** `{Phase}/tracking/loss_desc`.
- **Console:** `ld`.

**Computation**  
Mean $1-\cos$ between frozen query descriptors at $t{=}0$ and descriptors at predicted locations along the trajectory (`temporal_descriptor_consistency_loss`).

**Meaning**  
Agreement of the motion field with the **fixed** appearance embedding.

**Optimal trend**  
**Decrease**; large values may indicate coarse matcher drift the refinement net cannot fix.

---

### `loss_feat`

- **W&B:** `{Phase}/tracking/loss_feat`.
- **Console:** `lf`.

**Computation**  
Mean $\ell_2$ distance between fine features at the query location ($t{=}0$) and at tracked locations (`feature_persistence_loss`).

**Meaning**  
Feature-space persistence (more local than patch descriptors).

**Optimal trend**  
**Decrease**; spikes can indicate fast appearance change or bad pseudo masks on synthetic data.

---

### `loss_vis`

- **W&B:** `{Phase}/tracking/loss_vis`.
- **Console:** `lv`.

**Computation**  
Visibility **regularizer** only (not BCE to GT): flicker penalty on $\sigma(\ell)$ plus $0.1\times$ negative entropy (`visibility_regularization_loss`).

**Meaning**  
Encourages smooth, non-collapsed visibility logits.

**Optimal trend**  
Moderate **decrease** of flicker; entropy term prevents trivial all-on / all-off solutions.

---

### `loss_total` (inside `metrics`)

- **W&B:** `{Phase}/tracking/loss_total`.
- **Console:** `lt`.

**Computation**  
Detached float of the same tensor as `Loss` after blending.

**Meaning**  
Duplicate readout of the optimization scalar for CSV / tracking-prefixed tables.

**Optimal trend**  
Matches **`Loss`**.

---

### `pseudo_gt_active`

- **W&B:** `{Phase}/tracking/pseudo_gt_active`.
- **Console:** `pga`.

**Computation**  
$1.0$ if the pseudo-GT branch ran this step, else $0.0$.

**Meaning**  
Which rows of the log are hybrid-supervised vs. pure self-supervised.

**Optimal trend**  
N/A (indicator); use with `pseudo_lambda` to read curriculum phase.

---

### `pseudo_lambda`

- **W&B:** `{Phase}/tracking/pseudo_lambda`.
- **Console:** `pl`.

**Computation**  
$\lambda(e)$ defined above when pseudo ran; $0$ otherwise.

**Meaning**  
Instantaneous weight on supervised loss in the blend.

**Optimal trend**  
Ramps **up** to $\lambda_{\max}$ over the first $E_{\mathrm{cur}}$ epochs (for steps where pseudo is active).

---

### `loss_sup_pos`

- **W&B:** `{Phase}/tracking/loss_sup_pos` (only when pseudo supervised block ran).
- **Console:** `lsp`.

**Computation**  
Masked Smooth L1 on $\lVert \hat{\mathbf{p}} - \mathbf{p}^{\mathrm{gt}}\rVert_2$ normalized by mask mass.

**Meaning**  
Geometric error vs. renderer pseudo-GT.

**Optimal trend**  
**Decrease** when $\lambda > 0$.

---

### `loss_sup_vis`

- **W&B:** `{Phase}/tracking/loss_sup_vis`.
- **Console:** `lsv`.

**Computation**  
`binary_cross_entropy_with_logits` between $\ell$ and `vis_target` (mean reduction).

**Meaning**  
Visibility match to pseudo-GT (or appearance-aware variant).

**Optimal trend**  
**Decrease** when supervised mixing is on.

---

### `loss_sup_total`

- **W&B:** `{Phase}/tracking/loss_sup_total`.
- **Console:** `lst`.

**Computation**  
$w_{\mathrm{pos}}\mathcal{L}_{\mathrm{sup,pos}} + w_{\mathrm{vis}}\mathcal{L}_{\mathrm{sup,vis}}$.

**Meaning**  
Total supervised loss **before** blending with $\mathcal{L}_{\mathrm{self}}$.

**Optimal trend**  
**Decrease** during pseudo-active steps.

---

### `sup_mask_fraction`

- **W&B:** `{Phase}/tracking/sup_mask_fraction`.
- **Console:** `smf`.

**Computation**  
Mean of composite mask $M_{b,q,t}$.

**Meaning**  
How much spacetime area actually receives position supervision.

**Optimal trend**  
Stable fraction; **near zero** means almost no position supervision despite pseudo views.

---

### `epoch_loss`

- **W&B:** `{Phase}/tracking/epoch_loss` (logged with `step/epoch`).
- **Console:** — (printed separately as `>> {phase} epoch ... mean loss` text log).

**Computation**  
$\mathrm{mean}\bigl(\{\mathcal{L}_{\mathrm{total}}^{(\text{batch})}\}\bigr)$ over the epoch.

**Meaning**  
Epoch-level summary of the same scalar as `Loss`.

---

### StereoMIS: `delta_avg`

- **W&B:** `Validation/stereomis_gt/{sequence}/delta_avg`.

**Computation**  
TAP-Vid-style average, over thresholds $\Theta = \{1,2,4,8,16\}$ pixels, of visible-frame accuracy:

$$
\delta_{\mathrm{avg}} = \frac{1}{\lvert\Theta\rvert} \sum_{\theta \in \Theta} \frac{1}{\lvert\mathcal{V}\rvert} \sum_{(i,t)\in\mathcal{V}} \mathbf{1}\bigl[ \lVert \hat{\mathbf{p}}_i^t - \mathbf{p}_i^t \rVert_2 \lt \theta \bigr],
$$

with $\mathcal{V}$ the set of GT-visible space-time indices (`delta_avg` in `gatetracker/tracking/metrics.py`).

**Meaning**  
Real-world positional agreement on instrumented StereoMIS tracks (GT-initialized `track_long_sequence`).

**Optimal trend**  
**Increase** toward $1$.

---

### StereoMIS: `OA` (occlusion accuracy)

- **W&B:** `Validation/stereomis_gt/{sequence}/OA`.

**Computation**  
Classification accuracy of predicted visible vs. GT visible flags (`occlusion_accuracy`).

**Meaning**  
Binary visibility correctness.

**Optimal trend**  
**Increase**.

---

### StereoMIS: `AJ` (average Jaccard)

- **W&B:** `Validation/stereomis_gt/{sequence}/AJ`.

**Computation**  
Mean over $\theta \in \Theta$ of Jaccard index built from TP/FP/FN on “predicted visible AND within $\theta$ AND GT visible” (`average_jaccard`).

**Meaning**  
Joint quality of **position** (within threshold) and **visibility** on visible GT frames.

**Optimal trend**  
**Increase**.

---

### StereoMIS: videos

- **W&B:** `Validation/stereomis_gt/grid_video`, `Validation/stereomis_gt/gt_vs_pred_video` (when rendering succeeds).

**Meaning**  
Qualitative inspection: regular grid tracks vs. dataset frames, and GT vs. prediction overlay with error coloring.

---

## Appendix (pretrain only): triplet descriptor loss (`DESCRIPTOR_LOSS: triplet`)

If `config` sets `DESCRIPTOR_LOSS` to `triplet` (not the pretrain default), `DescriptorLoss` comes from `TripletLoss` in `gatetracker/utils/engine_init.py` / `losses` on mined anchors $\mathbf{a}$, positives $\mathbf{p}$, negatives $\mathbf{n}$ (typically margin-based on cosine or Euclidean distance). **W&B keys and all other pretrain metrics in this document are unchanged**; only the interpretation of `DescriptorLoss` switches from InfoNCE cross-entropy to margin triplet energy.
