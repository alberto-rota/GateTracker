"""
TAP-Vid evaluation metrics for point tracking.

All functions are fully vectorized (no loops over points/frames) and operate
on GPU tensors.

Metrics implemented:
- delta_avg: position accuracy averaged over pixel thresholds {1,2,4,8,16}
- OA:        occlusion (visibility) classification accuracy
- AJ:        average Jaccard index over the same thresholds
"""

import torch
from typing import Dict


_DEFAULT_THRESHOLDS = (1.0, 2.0, 4.0, 8.0, 16.0)

# STIR Challenge 2D accuracy thresholds (pixels, native 1280x1024 grid).
STIR_THRESHOLDS_PX = (4.0, 8.0, 16.0, 32.0, 64.0)


def position_accuracy(
    pred: torch.Tensor,
    gt: torch.Tensor,
    gt_vis: torch.Tensor,
    thresholds: tuple[float, ...] = _DEFAULT_THRESHOLDS,
) -> torch.Tensor:
    r"""
    Per-threshold fraction of visible points tracked within threshold.

    .. math::
        \delta^{\theta} = \frac{1}{|\mathcal{V}|}
        \sum_{(i,t) \in \mathcal{V}} \mathbf{1}[\|\hat{p}_i^t - p_i^t\|_2 < \theta]

    Args:
        pred:   [N, T, 2] predicted positions.
        gt:     [N, T, 2] ground-truth positions.
        gt_vis: [N, T]    ground-truth visibility (bool or 0/1).
        thresholds: pixel thresholds.

    Returns:
        [len(thresholds)] accuracy per threshold.
    """
    vis = gt_vis.bool()  # [N, T]
    dist = (pred - gt).norm(dim=-1)  # [N, T]
    n_visible = vis.sum().clamp_min(1).float()

    thresholds_t = torch.tensor(thresholds, device=pred.device, dtype=pred.dtype)  # [Θ]
    within = dist.unsqueeze(-1) < thresholds_t.view(1, 1, -1)  # [N, T, Θ]
    correct = within & vis.unsqueeze(-1)  # [N, T, Θ]
    return correct.sum(dim=(0, 1)).float() / n_visible  # [Θ]


def delta_avg(
    pred: torch.Tensor,
    gt: torch.Tensor,
    gt_vis: torch.Tensor,
    thresholds: tuple[float, ...] = _DEFAULT_THRESHOLDS,
) -> float:
    r"""
    Average position accuracy over thresholds:

    .. math::
        \delta_{\text{avg}} = \frac{1}{|\Theta|} \sum_{\theta} \delta^{\theta}
    """
    return float(position_accuracy(pred, gt, gt_vis, thresholds).mean())


def occlusion_accuracy(
    pred_vis: torch.Tensor,
    gt_vis: torch.Tensor,
) -> float:
    """Binary classification accuracy of visibility predictions vs GT."""
    pred_bool = pred_vis.bool()
    gt_bool = gt_vis.bool()
    return float((pred_bool == gt_bool).float().mean())


def average_jaccard(
    pred: torch.Tensor,
    gt: torch.Tensor,
    gt_vis: torch.Tensor,
    pred_vis: torch.Tensor,
    thresholds: tuple[float, ...] = _DEFAULT_THRESHOLDS,
) -> float:
    r"""
    Average Jaccard index over pixel thresholds.

    .. math::
        \text{AJ} = \frac{1}{|\Theta|}
        \sum_{\theta} \frac{\text{TP}_\theta}
        {\text{TP}_\theta + \text{FP}_\theta + \text{FN}_\theta}

    where:
    - TP = predicted visible AND within threshold AND GT visible
    - FP = predicted visible AND (NOT within threshold OR NOT GT visible)
    - FN = GT visible AND NOT predicted visible
    """
    vis_gt = gt_vis.bool()        # [N, T]
    vis_pred = pred_vis.bool()    # [N, T]
    dist = (pred - gt).norm(dim=-1)  # [N, T]

    thresholds_t = torch.tensor(thresholds, device=pred.device, dtype=pred.dtype)  # [Θ]
    within = dist.unsqueeze(-1) < thresholds_t.view(1, 1, -1)  # [N, T, Θ]

    tp = (vis_pred.unsqueeze(-1) & within & vis_gt.unsqueeze(-1)).sum(dim=(0, 1)).float()  # [Θ]
    fp = (vis_pred.unsqueeze(-1) & ~(within & vis_gt.unsqueeze(-1))).sum(dim=(0, 1)).float()  # [Θ]
    fn = (vis_gt.unsqueeze(-1) & ~vis_pred.unsqueeze(-1)).sum(dim=(0, 1)).float()  # [Θ]

    jaccard = tp / (tp + fp + fn).clamp_min(1.0)  # [Θ]
    return float(jaccard.mean())


def compute_stir_endpoint_metrics(
    pred_end: torch.Tensor,
    gt_end: torch.Tensor,
    thresholds: tuple[float, ...] = STIR_THRESHOLDS_PX,
) -> Dict[str, float]:
    r"""STIR Challenge 2D accuracy (unidirectional nearest-neighbour).

    For every predicted endpoint we take the distance to the *nearest* GT end
    center (unmatched sets) and report the fraction of predictions that fall
    below each threshold plus their mean:

    .. math::
        d_i = \min_{j} \| \hat{p}_i - g_j \|_2, \qquad
        \delta^{\theta} = \frac{1}{N}\sum_{i} \mathbf{1}[d_i < \theta], \qquad
        \delta_{\text{avg}} = \frac{1}{|\Theta|}\sum_{\theta \in \Theta}\delta^{\theta}.

    Both ``pred_end`` and ``gt_end`` must live in the **same pixel grid** (the
    STIR thresholds ``(4, 8, 16, 32, 64)`` are defined in the native
    1280x1024 segmentation grid).

    Args:
        pred_end: ``[N, 2]`` predicted endpoints.
        gt_end:   ``[M, 2]`` GT tattoo centers at the last frame.
        thresholds: pixel thresholds for accuracy.

    Returns:
        Dict with ``delta_avg``, ``mean_dist_px``, ``median_dist_px``,
        ``num_query_points``, ``num_gt_points``, and per-threshold
        ``acc_<th>px`` entries.
    """
    if pred_end.ndim != 2 or pred_end.shape[-1] != 2:
        raise ValueError(f"pred_end must be [N, 2], got {tuple(pred_end.shape)}")
    if gt_end.ndim != 2 or gt_end.shape[-1] != 2:
        raise ValueError(f"gt_end must be [M, 2], got {tuple(gt_end.shape)}")

    n = int(pred_end.shape[0])
    m = int(gt_end.shape[0])
    out: Dict[str, float] = {
        "num_query_points": float(n),
        "num_gt_points": float(m),
    }
    if n == 0 or m == 0:
        out["delta_avg"] = float("nan")
        out["mean_dist_px"] = float("nan")
        out["median_dist_px"] = float("nan")
        for th in thresholds:
            out[f"acc_{int(th)}px"] = float("nan")
        return out

    pred = pred_end.detach().float()  # [N, 2]
    gt = gt_end.detach().float().to(pred.device)  # [M, 2]
    d_nn = torch.cdist(pred.unsqueeze(0), gt.unsqueeze(0)).squeeze(0).min(dim=1).values  # [N]

    th_t = torch.tensor(thresholds, dtype=pred.dtype, device=pred.device)  # [Θ]
    accs = (d_nn.unsqueeze(-1) < th_t.view(1, -1)).float().mean(dim=0)  # [Θ]

    out["delta_avg"] = float(accs.mean())
    out["mean_dist_px"] = float(d_nn.mean())
    out["median_dist_px"] = float(d_nn.median())
    for th, a in zip(thresholds, accs.tolist()):
        out[f"acc_{int(th)}px"] = float(a)
    return out


def compute_tap_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    gt_vis: torch.Tensor,
    pred_vis: torch.Tensor,
    thresholds: tuple[float, ...] = _DEFAULT_THRESHOLDS,
) -> Dict[str, float]:
    """
    Compute all TAP-Vid metrics in one call.

    Args:
        pred:     [N, T, 2] predicted positions.
        gt:       [N, T, 2] ground-truth positions.
        gt_vis:   [N, T]    ground-truth visibility.
        pred_vis: [N, T]    predicted visibility.
        thresholds: pixel thresholds for delta_avg and AJ.

    Returns:
        Dict with keys ``delta_avg``, ``OA``, ``AJ``.
    """
    return {
        "delta_avg": delta_avg(pred, gt, gt_vis, thresholds),
        "OA": occlusion_accuracy(pred_vis, gt_vis),
        "AJ": average_jaccard(pred, gt, gt_vis, pred_vis, thresholds),
    }
