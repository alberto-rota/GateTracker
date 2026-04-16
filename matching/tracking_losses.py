"""
Self-supervised tracking losses and metrics computation.

All losses are differentiable and do not require ground-truth tracks.

INTEGRATION INTO engine.py
==========================
In the training step, after the matching forward pass:

    from matching.tracking_losses import (
        compute_tracking_losses,
        compute_tracking_metrics,
    )

    if config.get("TRACKING_MODE", False) and config.get("TRACKING_LOSS_WEIGHT", 0.0) > 0:
        tracking_result = compute_tracking_losses(
            matching_pipeline=self.matcher,
            source_image=source,
            target_image=target,
            model_output=modeloutput,
            config=self.config,
        )
        total_loss = total_loss + tracking_result["loss_total"]

        # Log scalar metrics
        for key, val in tracking_result["metrics"].items():
            metrics[f"track/{key}"] = val

In the validation / logging step (at LOG_FREQ cadence):

    tracking_metrics = compute_tracking_metrics(
        matching_pipeline=self.matcher,
        source_image=source,
        target_image=target,
        model_output=modeloutput,
        config=self.config,
    )
    for key, val in tracking_metrics.items():
        metrics[f"track/{key}"] = val

Then pass ``metrics`` dict to ``wandb.log(metrics, step=step)`` and
``logger.info(...)`` as usual.
"""

import torch
import torch.nn.functional as F

from matching import correspondence


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
    from utilities.tensor_utils import embedding2chw

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


def compute_tracking_losses(
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
        "loss_total": loss_total.item(),
        "cycle_error": cycle_err,
        "coarse_to_fine_delta": coarse_fine_delta,
        "confidence_mean": scores_fwd.mean().item(),
        "confidence_std": scores_fwd.std().item(),
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


def compute_tracking_metrics(
    matching_pipeline,
    source_image: torch.Tensor,
    target_image: torch.Tensor,
    model_output: dict,
    config,
) -> dict:
    """
    Compute tracking evaluation metrics (no gradient needed).

    Same as compute_tracking_losses but purely for logging; uses torch.no_grad.

    Returns:
        flat dict of ``track/*`` metric values (floats).
    """
    with torch.no_grad():
        result = compute_tracking_losses(
            matching_pipeline, source_image, target_image, model_output, config,
        )

    from utilities.tensor_utils import embedding2chw

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
