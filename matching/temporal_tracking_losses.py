"""
Self-supervised temporal tracking losses for multi-frame sequences.

All losses operate on temporal windows [B, Q, T, 2] and do **not** require
ground-truth tracks. They provide supervision signal for the
TemporalRefinementNetwork through geometric and appearance consistency.
"""

import torch
import torch.nn.functional as F

from matching import correspondence


# ---------------------------------------------------------------------------
# 1. Multi-frame cycle consistency
# ---------------------------------------------------------------------------

def temporal_cycle_consistency_loss(
    tracks: torch.Tensor,
    reverse_tracks: torch.Tensor,
    query_points: torch.Tensor,
    visibility: torch.Tensor | None = None,
) -> torch.Tensor:
    """Penalize round-trip error: track forward T frames, track backward, compare.

    Additionally penalizes waypoint consistency at intermediate frames.

    Args:
        tracks:         [B, Q, T, 2] forward tracked positions.
        reverse_tracks: [B, Q, T, 2] backward tracked positions (reversed time).
        query_points:   [B, Q, 2] original positions at t=0.
        visibility:     [B, Q, T] predicted visibility logits (optional).

    Returns:
        Scalar loss.
    """
    # Endpoint cycle error: roundtrip position at t=0 vs original query
    roundtrip_t0 = reverse_tracks[:, :, 0, :]  # [B, Q, 2]
    endpoint_err = (query_points - roundtrip_t0).abs().sum(dim=-1)  # [B, Q]

    # Intermediate waypoint consistency: fwd position at t should match
    # backward position at corresponding index
    T = tracks.shape[2]
    if T > 2:
        # Compare intermediate frames (exclude endpoints)
        mid_err = (tracks[:, :, 1:-1, :] - reverse_tracks[:, :, 1:-1, :]).abs().sum(dim=-1)  # [B, Q, T-2]
        mid_err = mid_err.mean(dim=-1)  # [B, Q]
    else:
        mid_err = torch.zeros_like(endpoint_err)

    err = endpoint_err + 0.5 * mid_err  # [B, Q]

    if visibility is not None:
        vis_prob = torch.sigmoid(visibility[:, :, 0])  # [B, Q]
        err = err * vis_prob.detach()

    return err.mean()


# ---------------------------------------------------------------------------
# 2. Temporal smoothness (acceleration penalty)
# ---------------------------------------------------------------------------

def temporal_smoothness_loss(
    tracks: torch.Tensor,
    visibility: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Second-order finite-difference penalty on track positions.

    .. math::
        \mathcal{L}_{\text{smooth}} = \frac{1}{BQ(T-2)}
            \sum_{b,q,t} \| p_{t+1} - 2 p_t + p_{t-1} \|_2

    Args:
        tracks:     [B, Q, T, 2] tracked positions.
        visibility: [B, Q, T] visibility logits (optional).

    Returns:
        Scalar loss.
    """
    T = tracks.shape[2]
    if T < 3:
        return torch.tensor(0.0, device=tracks.device, dtype=tracks.dtype)

    accel = tracks[:, :, 2:, :] - 2 * tracks[:, :, 1:-1, :] + tracks[:, :, :-2, :]  # [B, Q, T-2, 2]
    accel_norm = accel.norm(dim=-1)  # [B, Q, T-2]

    if visibility is not None:
        # Use minimum visibility across the three involved frames
        vis_prob = torch.sigmoid(visibility)
        v0 = vis_prob[:, :, :-2]
        v1 = vis_prob[:, :, 1:-1]
        v2 = vis_prob[:, :, 2:]
        weight = torch.min(torch.min(v0, v1), v2).detach()  # [B, Q, T-2]
        accel_norm = accel_norm * weight

    return accel_norm.mean()


# ---------------------------------------------------------------------------
# 3. Descriptor consistency across time
# ---------------------------------------------------------------------------

def temporal_descriptor_consistency_loss(
    tracks: torch.Tensor,
    query_points: torch.Tensor,
    descriptor_maps: list[torch.Tensor],
    patch_size: int,
    visibility: torch.Tensor | None = None,
) -> torch.Tensor:
    """Penalize descriptor drift: query descriptor should match target descriptor
    at tracked position across all T frames.

    Args:
        tracks:          [B, Q, T, 2] tracked positions.
        query_points:    [B, Q, 2] original positions at t=0.
        descriptor_maps: List of T tensors, each [B, C, H_p, W_p].
        patch_size:      Patch size for descriptor sampling.
        visibility:      [B, Q, T] visibility logits (optional).

    Returns:
        Scalar loss.
    """
    B, Q, T, _ = tracks.shape

    # Query descriptor from frame 0
    query_desc = correspondence.sample_embeddings_at_points(
        descriptor_maps[0], query_points, patch_size,
    )  # [B, Q, C]

    cos_dists = []
    for t in range(T):
        target_desc = correspondence.sample_embeddings_at_points(
            descriptor_maps[t], tracks[:, :, t, :], patch_size,
        )  # [B, Q, C]
        cos_sim = (query_desc * target_desc).sum(dim=-1)  # [B, Q]
        cos_dists.append(1.0 - cos_sim)

    cos_dist_stack = torch.stack(cos_dists, dim=-1)  # [B, Q, T]

    if visibility is not None:
        vis_prob = torch.sigmoid(visibility).detach()
        cos_dist_stack = cos_dist_stack * vis_prob

    return cos_dist_stack.mean()


# ---------------------------------------------------------------------------
# 4. Feature persistence loss
# ---------------------------------------------------------------------------

def feature_persistence_loss(
    tracks: torch.Tensor,
    query_points: torch.Tensor,
    fine_feature_maps: torch.Tensor,
    feature_stride: int,
    visibility: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fine feature at tracked location should remain similar to query's fine feature.

    More robust than photometric loss in surgical scenes with lighting changes.

    Args:
        tracks:            [B, Q, T, 2] tracked positions.
        query_points:      [B, Q, 2] original positions at t=0.
        fine_feature_maps: [B, T, C_f, H_f, W_f] fine feature maps.
        feature_stride:    Pixel stride of the fine feature map.
        visibility:        [B, Q, T] visibility logits (optional).

    Returns:
        Scalar loss.
    """
    B, Q, T, _ = tracks.shape
    C_f = fine_feature_maps.shape[2]

    # Query fine features at t=0
    query_fine = correspondence.sample_embeddings_at_points(
        fine_feature_maps[:, 0], query_points, feature_stride,
    )  # [B, Q, C_f]

    diffs = []
    for t in range(T):
        tracked_fine = correspondence.sample_embeddings_at_points(
            fine_feature_maps[:, t], tracks[:, :, t, :], feature_stride,
        )  # [B, Q, C_f]
        diff = (query_fine - tracked_fine).norm(dim=-1)  # [B, Q]
        diffs.append(diff)

    diff_stack = torch.stack(diffs, dim=-1)  # [B, Q, T]

    if visibility is not None:
        vis_prob = torch.sigmoid(visibility).detach()
        diff_stack = diff_stack * vis_prob

    return diff_stack.mean()


# ---------------------------------------------------------------------------
# 5. Visibility regularization
# ---------------------------------------------------------------------------

def visibility_regularization_loss(
    visibility: torch.Tensor,
) -> torch.Tensor:
    """Encourage temporally smooth and non-degenerate visibility predictions.

    Combines temporal smoothness (penalize flickering) with entropy
    regularization (avoid all-visible collapse).

    Args:
        visibility: [B, Q, T] visibility logits.

    Returns:
        Scalar loss.
    """
    vis_prob = torch.sigmoid(visibility)  # [B, Q, T]
    T = vis_prob.shape[2]

    # Temporal smoothness: penalize frame-to-frame visibility changes
    if T >= 2:
        flicker = (vis_prob[:, :, 1:] - vis_prob[:, :, :-1]).abs()  # [B, Q, T-1]
        smooth_loss = flicker.mean()
    else:
        smooth_loss = torch.tensor(0.0, device=visibility.device)

    # Binary entropy regularization to prevent all-visible collapse
    eps = 1e-6
    entropy = -(
        vis_prob * (vis_prob + eps).log()
        + (1 - vis_prob) * (1 - vis_prob + eps).log()
    )  # [B, Q, T]
    # Maximize entropy → minimize negative entropy
    entropy_loss = -entropy.mean()

    return smooth_loss + 0.1 * entropy_loss


# ---------------------------------------------------------------------------
# Combined loss computation
# ---------------------------------------------------------------------------

def compute_temporal_tracking_losses(
    tracks: torch.Tensor,
    reverse_tracks: torch.Tensor,
    query_points: torch.Tensor,
    visibility: torch.Tensor,
    descriptor_maps: list[torch.Tensor],
    fine_feature_maps: torch.Tensor,
    patch_size: int,
    feature_stride: int,
    config,
) -> dict:
    """Compute all self-supervised temporal tracking losses.

    Args:
        tracks:            [B, Q, T, 2] forward refined tracks.
        reverse_tracks:    [B, Q, T, 2] backward refined tracks.
        query_points:      [B, Q, 2] initial query positions.
        visibility:        [B, Q, T] visibility logits from refinement net.
        descriptor_maps:   List of T tensors [B, C, H_p, W_p] (coarse descriptors).
        fine_feature_maps: [B, T, C_f, H_f, W_f] fine feature maps.
        patch_size:        Patch size for coarse descriptor grid.
        feature_stride:    Pixel stride of fine features.
        config:            DotMap config.

    Returns:
        dict with ``loss_total`` (differentiable scalar) and ``metrics`` sub-dict.
    """
    w_cycle = float(
        config.get(
            "_SCHED_CYCLE_LOSS_WEIGHT",
            config.get("TEMPORAL_CYCLE_LOSS_WEIGHT", 1.0),
        )
    )
    w_smooth = float(
        config.get(
            "_SCHED_SMOOTHNESS_LOSS_WEIGHT",
            config.get("TEMPORAL_SMOOTHNESS_LOSS_WEIGHT", 0.3),
        )
    )
    w_desc = float(config.get("TEMPORAL_DESC_LOSS_WEIGHT", 0.5))
    w_feat = float(config.get("TEMPORAL_FEATURE_LOSS_WEIGHT", 0.5))
    w_vis = float(config.get("TEMPORAL_VIS_REG_WEIGHT", 0.1))

    loss_cycle = temporal_cycle_consistency_loss(
        tracks, reverse_tracks, query_points, visibility,
    )
    loss_smooth = temporal_smoothness_loss(tracks, visibility)
    loss_desc = temporal_descriptor_consistency_loss(
        tracks, query_points, descriptor_maps, patch_size, visibility,
    )
    loss_feat = feature_persistence_loss(
        tracks, query_points, fine_feature_maps, feature_stride, visibility,
    )
    loss_vis = visibility_regularization_loss(visibility)

    loss_cycle = torch.nan_to_num(loss_cycle, nan=0.0)
    loss_smooth = torch.nan_to_num(loss_smooth, nan=0.0)
    loss_desc = torch.nan_to_num(loss_desc, nan=0.0)
    loss_feat = torch.nan_to_num(loss_feat, nan=0.0)
    loss_vis = torch.nan_to_num(loss_vis, nan=0.0)

    loss_total = (
        w_cycle * loss_cycle
        + w_smooth * loss_smooth
        + w_desc * loss_desc
        + w_feat * loss_feat
        + w_vis * loss_vis
    )

    metrics = {
        "loss_cycle": loss_cycle.item(),
        "loss_smooth": loss_smooth.item(),
        "loss_desc": loss_desc.item(),
        "loss_feat": loss_feat.item(),
        "loss_vis": loss_vis.item(),
        "loss_total": loss_total.item(),
    }

    return {
        "loss_total": loss_total,
        "metrics": metrics,
    }
