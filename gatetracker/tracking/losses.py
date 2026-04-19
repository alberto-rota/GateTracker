"""
Unified tracking losses for GateTracker.

Contains both pair-frame tracking losses (from the matching pipeline) and
multi-frame temporal losses (for the TemporalRefinementNetwork).

All losses are differentiable and do not require ground-truth tracks.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from gatetracker.matching import correspondence
from gatetracker.utils.tensor_ops import embedding2chw


# ===================================================================
# Shared helpers
# ===================================================================

def validity_at_tracks_bqt(
    frame_valid_bt1hw: torch.Tensor,
    tracks_bqt2: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Sample per-frame validity (holemask) at subpixel track locations.

    Args:
        frame_valid_bt1hw: [B, T, 1, H, W] values in [0, 1].
        tracks_bqt2:       [B, Q, T, 2] pixel coords (x, y) = (u, v).
        height, width:     Image size H, W.

    Returns:
        w_rgb: [B, Q, T] sampled validity in [0, 1].
    """
    B, T, _, H, W = frame_valid_bt1hw.shape
    _, Q, Tt, _ = tracks_bqt2.shape
    assert T == Tt, "frame_valid and tracks must agree on T"

    fv = frame_valid_bt1hw.reshape(B * T, 1, H, W)
    uv = tracks_bqt2.permute(0, 2, 1, 3).reshape(B * T, Q, 2)
    gx = 2.0 * uv[..., 0].clamp(0.0, float(max(width - 1, 1))) / float(max(width - 1, 1)) - 1.0
    gy = 2.0 * uv[..., 1].clamp(0.0, float(max(height - 1, 1))) / float(max(height - 1, 1)) - 1.0
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(2)  # [BT, Q, 1, 2]
    sampled = F.grid_sample(
        fv, grid, mode="bilinear", padding_mode="zeros", align_corners=True,
    )  # [BT, 1, Q, 1]
    w = sampled.squeeze(1).squeeze(-1).reshape(B, T, Q).permute(0, 2, 1)  # [B, Q, T]
    return w.clamp(0.0, 1.0)


def in_bounds_mask_bqt(tracks_bqt2: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """[B, Q, T] bool: track inside image."""
    u, v = tracks_bqt2[..., 0], tracks_bqt2[..., 1]
    return (
        (u >= 0.0)
        & (u < float(width))
        & (v >= 0.0)
        & (v < float(height))
    )


def composite_supervision_mask(
    vis_gt_bqt: torch.Tensor,
    w_rgb_bqt: torch.Tensor,
    tracks_gt_bqt2: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """M[b,q,t] = vis_gt & w_rgb & in_bounds (float 0/1)."""
    if vis_gt_bqt.dtype == torch.bool:
        vis = vis_gt_bqt
    else:
        vis = vis_gt_bqt > 0.5
    ib = in_bounds_mask_bqt(tracks_gt_bqt2, height, width)
    m = vis & (w_rgb_bqt > 0.5) & ib
    return m.to(dtype=tracks_gt_bqt2.dtype)


def temporal_supervised_losses(
    tracks_pred: torch.Tensor,
    visibility_pred: torch.Tensor,
    tracks_gt: torch.Tensor,
    composite_mask: torch.Tensor,
    vis_target: torch.Tensor,
    config,
) -> dict:
    """Masked Huber position + BCE visibility vs pseudo-GT.

    Args:
        tracks_pred:      [B, Q, T, 2]
        visibility_pred:  [B, Q, T] logits
        tracks_gt:        [B, Q, T, 2]
        composite_mask:   [B, Q, T] bool/float — position supervision mask.
        vis_target:       [B, Q, T] float in [0,1] for BCE (appearance-aware or strict).
        config:           DotMap.

    Returns:
        dict with ``loss_sup_total``, ``loss_sup_pos``, ``loss_sup_vis``, ``metrics``.
    """
    huber_beta = float(config.get("TEMPORAL_SUP_HUBER_BETA", 1.0))
    eps = 1e-6

    mpos = composite_mask.to(dtype=tracks_pred.dtype)  # [B, Q, T]
    diff = tracks_pred - tracks_gt
    dist = diff.norm(dim=-1)  # [B, Q, T]
    loss_h = F.smooth_l1_loss(
        dist, torch.zeros_like(dist), beta=huber_beta, reduction="none",
    )
    loss_pos = (loss_h * mpos).sum() / (mpos.sum().clamp_min(eps))

    vt = vis_target.to(dtype=tracks_pred.dtype).clamp(0.0, 1.0)
    # Class-imbalanced BCE: occluded frames are rarer; up-weight negatives so the
    # head does not collapse to "always visible" under the position losses.
    w_occ = float(config.get("TEMPORAL_SUP_VIS_OCC_WEIGHT", 1.0))
    w_vis_el = 1.0 + (w_occ - 1.0) * (1.0 - vt)  # [B, Q, T]
    bce = F.binary_cross_entropy_with_logits(
        visibility_pred, vt, reduction="none",
    )
    loss_vis = (bce * w_vis_el).mean()

    w_pos = float(config.get("TEMPORAL_SUP_POS_WEIGHT", 1.0))
    w_vis = float(config.get("TEMPORAL_SUP_VIS_WEIGHT", 1.0))
    loss_sup_total = w_pos * loss_pos + w_vis * loss_vis

    frac_sup = float(mpos.mean().item())
    metrics = {
        "loss_sup_pos": loss_pos.item(),
        "loss_sup_vis": loss_vis.item(),
        "loss_sup_total": loss_sup_total.item(),
        "sup_mask_fraction": frac_sup,
    }
    return {
        "loss_sup_total": loss_sup_total,
        "loss_sup_pos": loss_pos,
        "loss_sup_vis": loss_vis,
        "metrics": metrics,
    }


def _sample_query_grid(B: int, H: int, W: int, num_points: int, device: torch.device) -> torch.Tensor:
    """Sample a regular grid of query points covering the image, with jitter.

    Returns:
        [B, Q, 2] pixel coordinates (x, y).
    """
    grid_side = max(1, int(num_points ** 0.5))
    Q = grid_side * grid_side
    margin_x = W / (2 * grid_side)
    margin_y = H / (2 * grid_side)
    xs = torch.linspace(margin_x, W - margin_x, grid_side, device=device)
    ys = torch.linspace(margin_y, H - margin_y, grid_side, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid_pts = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)  # [Q, 2]
    grid_pts = grid_pts.unsqueeze(0).expand(B, -1, -1).clone()  # [B, Q, 2]
    jitter_x = (W / grid_side) * 0.25
    jitter_y = (H / grid_side) * 0.25
    grid_pts[..., 0] += torch.empty(B, Q, device=device).uniform_(-jitter_x, jitter_x)
    grid_pts[..., 1] += torch.empty(B, Q, device=device).uniform_(-jitter_y, jitter_y)
    grid_pts[..., 0].clamp_(0, W - 1)
    grid_pts[..., 1].clamp_(0, H - 1)
    return grid_pts  # [B, Q, 2]


# ===================================================================
# Pair-frame tracking losses
# ===================================================================

def cycle_consistency_loss(
    fwd_tracked: torch.Tensor,
    roundtrip: torch.Tensor,
    query_points: torch.Tensor,
    fwd_vis: torch.Tensor = None,
    bwd_vis: torch.Tensor = None,
) -> torch.Tensor:
    """L1 round-trip error weighted by minimum forward/backward visibility.

    Args:
        fwd_tracked:  [B, Q, 2] forward tracked positions (unused except shape).
        roundtrip:    [B, Q, 2] round-trip positions after fwd + bwd tracking.
        query_points: [B, Q, 2] original source positions.
        fwd_vis:      [B, Q, 1] or None, forward visibility logits.
        bwd_vis:      [B, Q, 1] or None, backward visibility logits.

    Returns:
        Scalar loss.
    """
    err = (query_points - roundtrip).abs().sum(dim=-1)  # [B, Q]
    if fwd_vis is not None and bwd_vis is not None:
        fwd_prob = torch.sigmoid(fwd_vis.squeeze(-1))  # [B, Q]
        bwd_prob = torch.sigmoid(bwd_vis.squeeze(-1))  # [B, Q]
        weight = torch.min(fwd_prob, bwd_prob).detach()  # [B, Q]
        err = err * weight
    return err.mean()


def descriptor_similarity_loss(
    matching_pipeline,
    model_output: dict,
    query_points: torch.Tensor,
    tracked_points: torch.Tensor,
    scores: torch.Tensor,
) -> torch.Tensor:
    """1 - cos_sim between source descriptor at query and target descriptor at tracked.

    Weighted by tracking confidence to down-weight occluded points.

    Args:
        matching_pipeline: Matcher instance.
        model_output:      dict with source/target embedding maps.
        query_points:      [B, Q, 2]
        tracked_points:    [B, Q, 2]
        scores:            [B, Q] confidence weights.

    Returns:
        Scalar loss.
    """
    patch_size = matching_pipeline.patch_size
    src_map = embedding2chw(model_output["source_embedding_match"], embed_dim_last=False)
    tgt_map = embedding2chw(model_output["target_embedding_match"], embed_dim_last=False)

    src_emb = correspondence.sample_embeddings_at_points(src_map, query_points, patch_size)  # [B, Q, C]
    tgt_emb = correspondence.sample_embeddings_at_points(tgt_map, tracked_points, patch_size)  # [B, Q, C]

    cos_sim = (src_emb * tgt_emb).sum(dim=-1)  # [B, Q]
    loss = (1.0 - cos_sim) * scores.detach().clamp(0, 1)
    return loss.mean()


def coarse_to_fine_regularization(
    coarse_target: torch.Tensor,
    tracked_points: torch.Tensor,
    margin: float = 32.0,
) -> torch.Tensor:
    """Penalise fine refinement deltas that exceed a margin (in pixels).

    Args:
        coarse_target:  [B, Q, 2]
        tracked_points: [B, Q, 2]
        margin:         max expected delta magnitude (pixels).

    Returns:
        Scalar loss.
    """
    delta = (tracked_points - coarse_target).norm(dim=-1)  # [B, Q]
    excess = F.relu(delta - margin)  # [B, Q]
    return excess.mean()


def compute_pairwise_tracking_losses(
    matching_pipeline,
    source_image: torch.Tensor,
    target_image: torch.Tensor,
    model_output: dict,
    config,
) -> dict:
    """
    Compute all self-supervised tracking losses for a single frame pair.

    Args:
        matching_pipeline: Matcher instance.
        source_image:      [B, 3, H, W]
        target_image:      [B, 3, H, W]
        model_output:      dict from MatcherModel.forward().
        config:            DotMap config.

    Returns:
        dict with ``loss_total`` (scalar) and ``metrics`` sub-dict.
    """
    B, _, H, W = source_image.shape
    num_pts = int(config.get("TRACKING_NUM_QUERY_POINTS", 64))
    device = source_image.device

    w_total = float(config.get("TRACKING_LOSS_WEIGHT", 0.0))
    w_cycle = float(config.get("TRACKING_CYCLE_LOSS_WEIGHT", 1.0))
    w_desc = float(config.get("TRACKING_DESC_SIM_LOSS_WEIGHT", 0.5))
    w_reg = float(config.get("TRACKING_REG_LOSS_WEIGHT", 0.1))
    reg_margin = float(config.get("TRACKING_REG_MARGIN", 32.0))

    query_pts = _sample_query_grid(B, H, W, num_pts, device)  # [B, Q, 2]

    fwd = matching_pipeline.track_points(query_pts, source_image, target_image)
    tracked_pts = fwd["tracked_points"]  # [B, Q, 2]
    coarse_tgt = fwd["coarse_target"]  # [B, Q, 2]
    scores_fwd = fwd["scores"]  # [B, Q]
    vis_fwd = fwd.get("visibility_logit")  # [B, Q, 1] or None

    bwd = matching_pipeline.track_points(tracked_pts, target_image, source_image)
    roundtrip = bwd["tracked_points"]  # [B, Q, 2]
    vis_bwd = bwd.get("visibility_logit")  # [B, Q, 1] or None

    loss_cycle = cycle_consistency_loss(tracked_pts, roundtrip, query_pts, vis_fwd, vis_bwd)
    loss_desc = descriptor_similarity_loss(
        matching_pipeline, model_output, query_pts, tracked_pts, scores_fwd,
    )
    loss_reg = coarse_to_fine_regularization(coarse_tgt, tracked_pts, margin=reg_margin)

    loss_cycle = torch.nan_to_num(loss_cycle, nan=0.0)
    loss_desc = torch.nan_to_num(loss_desc, nan=0.0)
    loss_reg = torch.nan_to_num(loss_reg, nan=0.0)

    loss_total = w_total * (w_cycle * loss_cycle + w_desc * loss_desc + w_reg * loss_reg)

    cycle_err = (query_pts - roundtrip).abs().sum(dim=-1).mean().item()
    coarse_fine_delta = (tracked_pts - coarse_tgt).norm(dim=-1).mean().item()

    metrics = {
        "loss_cycle": loss_cycle.item(),
        "loss_desc": loss_desc.item(),
        "loss_reg": loss_reg.item(),
        "cycle_error": cycle_err,
        "coarse_to_fine_delta": coarse_fine_delta,
        "confidence_mean": scores_fwd.mean().item(),
    }
    if vis_fwd is not None:
        vis_prob = torch.sigmoid(vis_fwd.squeeze(-1))
        metrics["visibility_ratio"] = (vis_prob > 0.5).float().mean().item()

    return {
        "loss_total": loss_total,
        "metrics": metrics,
        "fwd_result": fwd,
        "bwd_result": bwd,
        "query_points": query_pts,
    }


def compute_pairwise_tracking_metrics(
    matching_pipeline,
    source_image: torch.Tensor,
    target_image: torch.Tensor,
    model_output: dict,
    config,
) -> dict:
    """
    Compute tracking evaluation metrics (no gradient needed).

    Same as compute_pairwise_tracking_losses but purely for logging; uses torch.no_grad.

    Returns:
        flat dict of ``track/*`` metric values (floats).
    """
    with torch.no_grad():
        result = compute_pairwise_tracking_losses(
            matching_pipeline, source_image, target_image, model_output, config,
        )

    patch_size = matching_pipeline.patch_size
    src_map = embedding2chw(model_output["source_embedding_match"], embed_dim_last=False)
    tgt_map = embedding2chw(model_output["target_embedding_match"], embed_dim_last=False)

    query_pts = result["query_points"]
    tracked_pts = result["fwd_result"]["tracked_points"]

    with torch.no_grad():
        src_emb = correspondence.sample_embeddings_at_points(src_map, query_pts, patch_size)
        tgt_emb = correspondence.sample_embeddings_at_points(tgt_map, tracked_pts, patch_size)
        descriptor_sim = (src_emb * tgt_emb).sum(dim=-1).mean().item()

    metrics = result["metrics"]
    metrics["descriptor_sim"] = descriptor_sim
    return metrics


# ===================================================================
# Multi-frame temporal tracking losses
# ===================================================================

def temporal_cycle_consistency_loss(
    tracks: torch.Tensor,
    reverse_tracks: torch.Tensor,
    query_points: torch.Tensor,
    visibility: Optional[torch.Tensor] = None,
    self_sup_mask_bqt: Optional[torch.Tensor] = None,
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
        mid_err = (tracks[:, :, 1:-1, :] - reverse_tracks[:, :, 1:-1, :]).abs().sum(dim=-1)  # [B, Q, T-2]
        mid_err = mid_err.mean(dim=-1)  # [B, Q]
    else:
        mid_err = torch.zeros_like(endpoint_err)

    err = endpoint_err + 0.5 * mid_err  # [B, Q]

    if visibility is not None:
        vis_prob = torch.sigmoid(visibility[:, :, 0])  # [B, Q]
        err = err * vis_prob.detach()

    if self_sup_mask_bqt is not None:
        wq = self_sup_mask_bqt.mean(dim=2).clamp(0.0, 1.0).detach()  # [B, Q]
        err = err * wq

    return err.mean()


def temporal_velocity_huber_loss(
    tracks: torch.Tensor,
    visibility: Optional[torch.Tensor] = None,
    self_sup_mask_bqt: Optional[torch.Tensor] = None,
    beta: float = 4.0,
) -> torch.Tensor:
    r"""First-order Huber on consecutive displacements (penalises single-frame spikes).

    Complements ``temporal_smoothness_loss`` (second-order): a brief outlier jump
    has small acceleration if it returns the next frame, but produces a large
    velocity impulse here.

    Args:
        tracks:     [B, Q, T, 2] tracked positions.
        visibility: [B, Q, T] visibility logits (optional).
        self_sup_mask_bqt: Optional [B, Q, T] float mask.
        beta:       Huber threshold in **pixels** (L2 step norm).

    Returns:
        Scalar loss.
    """
    T = tracks.shape[2]
    if T < 2:
        return torch.tensor(0.0, device=tracks.device, dtype=tracks.dtype)

    step = (tracks[:, :, 1:, :] - tracks[:, :, :-1, :]).norm(dim=-1)  # [B, Q, T-1]
    loss_h = F.smooth_l1_loss(
        step, torch.zeros_like(step), beta=float(beta), reduction="none",
    )

    if visibility is not None:
        vis_prob = torch.sigmoid(visibility)
        w = torch.minimum(vis_prob[:, :, :-1], vis_prob[:, :, 1:]).detach()  # [B, Q, T-1]
        loss_h = loss_h * w

    if self_sup_mask_bqt is not None:
        m0 = self_sup_mask_bqt[:, :, :-1]
        m1 = self_sup_mask_bqt[:, :, 1:]
        wm = torch.minimum(m0, m1).detach()
        loss_h = loss_h * wm

    return loss_h.mean()


def temporal_smoothness_loss(
    tracks: torch.Tensor,
    visibility: Optional[torch.Tensor] = None,
    self_sup_mask_bqt: Optional[torch.Tensor] = None,
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

    if self_sup_mask_bqt is not None:
        m0 = self_sup_mask_bqt[:, :, :-2]
        m1 = self_sup_mask_bqt[:, :, 1:-1]
        m2 = self_sup_mask_bqt[:, :, 2:]
        wm = torch.minimum(torch.minimum(m0, m1), m2).detach()  # [B, Q, T-2]
        accel_norm = accel_norm * wm

    return accel_norm.mean()


def temporal_descriptor_consistency_loss(
    tracks: torch.Tensor,
    query_points: torch.Tensor,
    descriptor_maps: list[torch.Tensor],
    patch_size: int,
    visibility: Optional[torch.Tensor] = None,
    self_sup_mask_bqt: Optional[torch.Tensor] = None,
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

    if self_sup_mask_bqt is not None:
        m0 = self_sup_mask_bqt[:, :, :1].expand_as(cos_dist_stack)
        cos_dist_stack = cos_dist_stack * torch.minimum(
            m0, self_sup_mask_bqt.detach(),
        )

    return cos_dist_stack.mean()


def feature_persistence_loss(
    tracks: torch.Tensor,
    query_points: torch.Tensor,
    fine_feature_maps: torch.Tensor,
    feature_stride: int,
    visibility: Optional[torch.Tensor] = None,
    self_sup_mask_bqt: Optional[torch.Tensor] = None,
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

    if self_sup_mask_bqt is not None:
        m0 = self_sup_mask_bqt[:, :, :1].expand_as(diff_stack)
        diff_stack = diff_stack * torch.minimum(
            m0, self_sup_mask_bqt.detach(),
        )

    return diff_stack.mean()


def visibility_regularization_loss(
    visibility: torch.Tensor,
    entropy_scale: float = 0.1,
) -> torch.Tensor:
    """Encourage temporally smooth and non-degenerate visibility predictions.

    Combines temporal smoothness (penalize flickering) with mild entropy
    regularization (avoid all-visible collapse). ``entropy_scale`` defaults to
    the historical 0.1; lower values (e.g. 0.02--0.05) sharpen occlusion logits
    when supervised visibility is strong.

    Args:
        visibility:    [B, Q, T] visibility logits.
        entropy_scale: Multiplier on the entropy term.

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

    return smooth_loss + float(entropy_scale) * entropy_loss


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
    self_sup_mask_bqt: Optional[torch.Tensor] = None,
    cycle_weight_scale: float = 1.0,
    synth_self_sup_scale: float = 1.0,
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
        self_sup_mask_bqt: Optional ``[B,Q,T]`` validity for descriptor/feat/cycle/smooth on synthetic.
        cycle_weight_scale: Multiplier on cycle loss weight (use ``0`` without backward tracking).
        synth_self_sup_scale: Multiplier on cycle/smooth/desc/feat (not ``loss_vis`` reg).

    Returns:
        dict with ``loss_total`` (differentiable scalar) and ``metrics`` (detached
        per-term scalars; the blended total is logged separately as ``Loss``).
    """
    w_cycle = float(config.get("TEMPORAL_CYCLE_LOSS_WEIGHT", 1.0)) * float(cycle_weight_scale)
    w_smooth = float(config.get("TEMPORAL_SMOOTHNESS_LOSS_WEIGHT", 0.3))
    w_vel = float(config.get("TEMPORAL_VELOCITY_HUBER_WEIGHT", 0.0))
    vel_beta = float(config.get("TEMPORAL_VELOCITY_HUBER_BETA", 4.0))
    w_desc = float(config.get("TEMPORAL_DESC_LOSS_WEIGHT", 0.5))
    w_feat = float(config.get("TEMPORAL_FEATURE_LOSS_WEIGHT", 0.5))
    w_vis = float(config.get("TEMPORAL_VIS_REG_WEIGHT", 0.1))
    ent_scale = float(config.get("TEMPORAL_VIS_REG_ENTROPY_SCALE", 0.1))
    w_synth = float(synth_self_sup_scale)

    loss_cycle = temporal_cycle_consistency_loss(
        tracks, reverse_tracks, query_points, visibility, self_sup_mask_bqt,
    )
    loss_smooth = temporal_smoothness_loss(tracks, visibility, self_sup_mask_bqt)
    loss_vel = temporal_velocity_huber_loss(
        tracks, visibility, self_sup_mask_bqt, beta=vel_beta,
    )
    loss_desc = temporal_descriptor_consistency_loss(
        tracks, query_points, descriptor_maps, patch_size, visibility, self_sup_mask_bqt,
    )
    loss_feat = feature_persistence_loss(
        tracks, query_points, fine_feature_maps, feature_stride, visibility, self_sup_mask_bqt,
    )
    loss_vis = visibility_regularization_loss(visibility, entropy_scale=ent_scale)

    # Replace NaN/Inf with 0 so a single degenerate batch cannot poison the
    # parameter update; clipping downstream bounds the effective step anyway.
    _sanitize = lambda t: torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    loss_cycle = _sanitize(loss_cycle)
    loss_smooth = _sanitize(loss_smooth)
    loss_vel = _sanitize(loss_vel)
    loss_desc = _sanitize(loss_desc)
    loss_feat = _sanitize(loss_feat)
    loss_vis = _sanitize(loss_vis)

    loss_core = (
        w_cycle * loss_cycle
        + w_smooth * loss_smooth
        + w_vel * loss_vel
        + w_desc * loss_desc
        + w_feat * loss_feat
    )
    loss_total = w_synth * loss_core + w_vis * loss_vis

    metrics = {
        "loss_cycle": loss_cycle.item(),
        "loss_smooth": loss_smooth.item(),
        "loss_vel": loss_vel.item(),
        "loss_desc": loss_desc.item(),
        "loss_feat": loss_feat.item(),
        "loss_vis": loss_vis.item(),
    }

    return {
        "loss_total": loss_total,
        "metrics": metrics,
    }
