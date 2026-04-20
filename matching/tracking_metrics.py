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
    device = pred_vis.device
    pred_bool = pred_vis.bool()
    gt_bool = gt_vis.to(device).bool()
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


def compute_tracking_metrics(
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
    device, dtype = pred.device, pred.dtype
    gt = gt.to(device=device, dtype=dtype)
    gt_vis = gt_vis.to(device=device)
    pred_vis = pred_vis.to(device=device)
    return {
        "delta_avg": delta_avg(pred, gt, gt_vis, thresholds),
        "OA": occlusion_accuracy(pred_vis, gt_vis),
        "AJ": average_jaccard(pred, gt, gt_vis, pred_vis, thresholds),
    }
