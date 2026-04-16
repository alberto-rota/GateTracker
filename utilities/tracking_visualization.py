"""
Tracking visualizations for WandB logging.

All functions return PIL Images or numpy arrays suitable for ``wandb.Image``.

INTEGRATION INTO engine.py
==========================
At the image-logging cadence (``LOG_FREQ_WANDB``), after computing tracking
losses/metrics:

    from utilities.tracking_visualization import build_tracking_visualizations

    if step % config.LOG_FREQ_WANDB == 0 and not config.NO_WANDB:
        tracking_result = ...  # from compute_tracking_losses
        tracking_images = build_tracking_visualizations(
            source_image=source,
            target_image=target,
            tracking_result=tracking_result,
            model_output=modeloutput,
            matching_pipeline=self.matcher,
            config=self.config,
        )
        for name, img in tracking_images.items():
            wandb.log({f"tracking/{name}": wandb.Image(img)}, step=step)

    # Also log to console via logger
    logger.info(
        f"[Tracking] cycle_err={metrics['track/cycle_error']:.3f} "
        f"desc_sim={metrics['track/descriptor_sim']:.3f} "
        f"c2f_delta={metrics['track/coarse_to_fine_delta']:.2f} "
        f"conf={metrics['track/confidence_mean']:.3f} "
        f"vis_ratio={metrics.get('track/visibility_ratio', -1):.3f}"
    )
"""

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw


def _tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """[3, H, W] float tensor in [0,1] -> [H, W, 3] uint8 numpy."""
    img = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def _score_to_rgb(score: float) -> tuple:
    """Map score in [0,1] to (R, G, B) from red (0) to green (1)."""
    r = int(255 * (1.0 - score))
    g = int(255 * score)
    return (r, g, 40)


def _colormap_values(values: np.ndarray, cmap_name: str = "RdYlGn") -> np.ndarray:
    """Map normalized [0,1] values to RGB via matplotlib colormap. Returns [N, 3] uint8."""
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(values.clip(0, 1))  # [N, 4]
    return (rgba[:, :3] * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# D1: Track Lines Overlay
# ---------------------------------------------------------------------------

def viz_track_lines(
    source_image: torch.Tensor,
    target_image: torch.Tensor,
    query_points: torch.Tensor,
    tracked_points: torch.Tensor,
    scores: torch.Tensor,
    batch_idx: int = 0,
    max_points: int = 100,
) -> Image.Image:
    """Source and target side-by-side with colored lines connecting tracked points.

    Line color: green = high confidence, red = low confidence.

    Args:
        source_image:  [B, 3, H, W]
        target_image:  [B, 3, H, W]
        query_points:  [B, Q, 2] (x, y) in source
        tracked_points: [B, Q, 2] (x, y) in target
        scores:        [B, Q] confidence
        batch_idx:     which batch element to visualize
        max_points:    subsample if Q > max_points

    Returns:
        PIL Image.
    """
    src_np = _tensor_to_uint8(source_image[batch_idx])
    tgt_np = _tensor_to_uint8(target_image[batch_idx])
    H, W = src_np.shape[:2]

    canvas = np.concatenate([src_np, tgt_np], axis=1)  # [H, 2W, 3]
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)

    q = query_points[batch_idx].detach().cpu().numpy()  # [Q, 2]
    t = tracked_points[batch_idx].detach().cpu().numpy()
    s = scores[batch_idx].detach().cpu().numpy()
    Q = q.shape[0]

    if Q > max_points:
        idx = np.linspace(0, Q - 1, max_points).astype(int)
        q, t, s = q[idx], t[idx], s[idx]

    for i in range(q.shape[0]):
        sx, sy = float(q[i, 0]), float(q[i, 1])
        tx, ty = float(t[i, 0]) + W, float(t[i, 1])
        color = _score_to_rgb(float(s[i]))
        draw.line([(sx, sy), (tx, ty)], fill=color, width=1)
        draw.ellipse([sx - 2, sy - 2, sx + 2, sy + 2], fill=color)
        draw.ellipse([tx - 2, ty - 2, tx + 2, ty + 2], fill=color)

    return pil


# ---------------------------------------------------------------------------
# D2: Coarse vs Fine Refinement
# ---------------------------------------------------------------------------

def viz_coarse_vs_fine(
    target_image: torch.Tensor,
    coarse_target: torch.Tensor,
    tracked_points: torch.Tensor,
    scores: torch.Tensor,
    batch_idx: int = 0,
    max_points: int = 100,
) -> Image.Image:
    """Target frame with circles at coarse positions and arrows to fine positions.

    Arrow length = refinement displacement.

    Args:
        target_image:  [B, 3, H, W]
        coarse_target: [B, Q, 2] coarse (grid-snapped) positions
        tracked_points: [B, Q, 2] fine-refined positions
        scores:        [B, Q]
        batch_idx:     batch element index
        max_points:    subsample cap

    Returns:
        PIL Image.
    """
    tgt_np = _tensor_to_uint8(target_image[batch_idx])
    pil = Image.fromarray(tgt_np)
    draw = ImageDraw.Draw(pil)

    c = coarse_target[batch_idx].detach().cpu().numpy()
    f = tracked_points[batch_idx].detach().cpu().numpy()
    s = scores[batch_idx].detach().cpu().numpy()
    Q = c.shape[0]

    if Q > max_points:
        idx = np.linspace(0, Q - 1, max_points).astype(int)
        c, f, s = c[idx], f[idx], s[idx]

    for i in range(c.shape[0]):
        cx, cy = float(c[i, 0]), float(c[i, 1])
        fx, fy = float(f[i, 0]), float(f[i, 1])
        color = _score_to_rgb(float(s[i]))
        draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], outline=(100, 100, 255), width=1)
        draw.line([(cx, cy), (fx, fy)], fill=color, width=2)
        draw.ellipse([fx - 2, fy - 2, fx + 2, fy + 2], fill=color)

    return pil


# ---------------------------------------------------------------------------
# D3: Cycle Consistency Error Map
# ---------------------------------------------------------------------------

def viz_cycle_error(
    source_image: torch.Tensor,
    query_points: torch.Tensor,
    roundtrip_points: torch.Tensor,
    batch_idx: int = 0,
    max_points: int = 200,
) -> Image.Image:
    """Source frame with circles at query points, colored by round-trip error.

    Green = small error (consistent), red = large error.

    Args:
        source_image:    [B, 3, H, W]
        query_points:    [B, Q, 2]
        roundtrip_points: [B, Q, 2] after forward+backward tracking
        batch_idx:       batch element
        max_points:      subsample cap

    Returns:
        PIL Image.
    """
    src_np = _tensor_to_uint8(source_image[batch_idx])
    pil = Image.fromarray(src_np)
    draw = ImageDraw.Draw(pil)

    q = query_points[batch_idx].detach().cpu().numpy()
    r = roundtrip_points[batch_idx].detach().cpu().numpy()
    Q = q.shape[0]

    errs = np.linalg.norm(q - r, axis=-1)  # [Q]
    max_err = max(errs.max(), 1e-6)
    normed = 1.0 - (errs / max_err).clip(0, 1)  # 1 = good, 0 = bad

    if Q > max_points:
        idx = np.linspace(0, Q - 1, max_points).astype(int)
        q, errs, normed = q[idx], errs[idx], normed[idx]

    colors = _colormap_values(normed, "RdYlGn")

    for i in range(q.shape[0]):
        x, y = float(q[i, 0]), float(q[i, 1])
        radius = max(2, min(8, int(errs[i] / max_err * 8)))
        c = tuple(colors[i].tolist())
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=c, outline=c)

    return pil


# ---------------------------------------------------------------------------
# D4: Gated Layer Heatmap at Tracked Points
# ---------------------------------------------------------------------------

def viz_gated_layer_heatmap(
    source_image: torch.Tensor,
    query_points: torch.Tensor,
    gated_layer_maps: dict,
    layer_indices: torch.Tensor,
    patch_size: int,
    batch_idx: int = 0,
    max_points: int = 200,
) -> Image.Image:
    """Source frame with points colored by effective layer index from interpolated gates.

    Blue = shallow layers, red = deep layers.

    Args:
        source_image:    [B, 3, H, W]
        query_points:    [B, Q, 2]
        gated_layer_maps: dict with ``gate_weight_maps`` [L, B, 1, H_p, W_p].
        layer_indices:   [L] tensor of absolute layer indices.
        patch_size:      patch stride in pixels.
        batch_idx:       batch element.
        max_points:      subsample cap.

    Returns:
        PIL Image.
    """
    import torch.nn.functional as F

    src_np = _tensor_to_uint8(source_image[batch_idx])
    pil = Image.fromarray(src_np)
    draw = ImageDraw.Draw(pil)

    gate_maps = gated_layer_maps["gate_weight_maps"]  # [L, B, 1, H_p, W_p]
    L, B_g, _, H_p, W_p = gate_maps.shape
    img_h, img_w = H_p * patch_size, W_p * patch_size

    pts = query_points[batch_idx:batch_idx+1].clone().float()  # [1, Q, 2]
    Q = pts.shape[1]

    grid = pts.clone()
    grid[..., 0] = grid[..., 0] / max(img_w - 1, 1) * 2 - 1
    grid[..., 1] = grid[..., 1] / max(img_h - 1, 1) * 2 - 1
    grid = grid.unsqueeze(2)  # [1, Q, 1, 2]

    gate_batch = gate_maps[:, batch_idx:batch_idx+1, :, :, :]  # [L, 1, 1, H_p, W_p]
    gate_flat = gate_batch.reshape(L, 1, H_p, W_p)  # [L, 1, H_p, W_p]
    grid_L = grid.expand(L, -1, -1, -1)  # [L, Q, 1, 2]
    sampled = F.grid_sample(
        gate_flat, grid_L, mode="bilinear", align_corners=True,
    )  # [L, 1, Q, 1]
    weights = sampled.squeeze(-1).squeeze(1)  # [L, Q]

    layer_idx_float = layer_indices.float().to(weights.device)  # [L]
    effective = (weights * layer_idx_float.unsqueeze(1)).sum(dim=0)  # [Q]

    eff_np = effective.detach().cpu().numpy()
    lo, hi = float(layer_indices.min()), float(layer_indices.max())
    normed = ((eff_np - lo) / max(hi - lo, 1e-6)).clip(0, 1)

    q_np = pts[0].detach().cpu().numpy()
    if Q > max_points:
        idx = np.linspace(0, Q - 1, max_points).astype(int)
        q_np, normed = q_np[idx], normed[idx]

    colors = _colormap_values(normed, "coolwarm")

    for i in range(q_np.shape[0]):
        x, y = float(q_np[i, 0]), float(q_np[i, 1])
        c = tuple(colors[i].tolist())
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=c, outline=c)

    return pil


# ---------------------------------------------------------------------------
# D5: Visibility Prediction Overlay
# ---------------------------------------------------------------------------

def viz_visibility(
    target_image: torch.Tensor,
    tracked_points: torch.Tensor,
    visibility_logit: torch.Tensor,
    batch_idx: int = 0,
    max_points: int = 200,
) -> Image.Image:
    """Target frame with tracked points colored by visibility probability.

    Bright green = visible, red = occluded.

    Args:
        target_image:     [B, 3, H, W]
        tracked_points:   [B, Q, 2]
        visibility_logit: [B, Q, 1] raw logits.
        batch_idx:        batch element.
        max_points:       subsample cap.

    Returns:
        PIL Image.
    """
    tgt_np = _tensor_to_uint8(target_image[batch_idx])
    pil = Image.fromarray(tgt_np)
    draw = ImageDraw.Draw(pil)

    t = tracked_points[batch_idx].detach().cpu().numpy()
    vis_prob = torch.sigmoid(visibility_logit[batch_idx].squeeze(-1)).detach().cpu().numpy()
    Q = t.shape[0]

    if Q > max_points:
        idx = np.linspace(0, Q - 1, max_points).astype(int)
        t, vis_prob = t[idx], vis_prob[idx]

    colors = _colormap_values(vis_prob, "RdYlGn")

    for i in range(t.shape[0]):
        x, y = float(t[i, 0]), float(t[i, 1])
        c = tuple(colors[i].tolist())
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=c, outline=c)

    return pil


# ---------------------------------------------------------------------------
# D6: Descriptor Similarity at Tracked Points
# ---------------------------------------------------------------------------

def viz_descriptor_similarity(
    target_image: torch.Tensor,
    tracked_points: torch.Tensor,
    source_emb: torch.Tensor,
    target_emb: torch.Tensor,
    batch_idx: int = 0,
    max_points: int = 200,
) -> Image.Image:
    """Target frame with tracked points colored by cosine similarity.

    Green = high similarity, red = low similarity.

    Args:
        target_image:  [B, 3, H, W]
        tracked_points: [B, Q, 2]
        source_emb:    [B, Q, C] L2-normalized source embeddings.
        target_emb:    [B, Q, C] L2-normalized target embeddings at tracked locs.
        batch_idx:     batch element.
        max_points:    subsample cap.

    Returns:
        PIL Image.
    """
    tgt_np = _tensor_to_uint8(target_image[batch_idx])
    pil = Image.fromarray(tgt_np)
    draw = ImageDraw.Draw(pil)

    t = tracked_points[batch_idx].detach().cpu().numpy()
    sim = (source_emb[batch_idx] * target_emb[batch_idx]).sum(dim=-1)
    sim_np = sim.detach().cpu().clamp(0, 1).numpy()
    Q = t.shape[0]

    if Q > max_points:
        idx = np.linspace(0, Q - 1, max_points).astype(int)
        t, sim_np = t[idx], sim_np[idx]

    colors = _colormap_values(sim_np, "RdYlGn")

    for i in range(t.shape[0]):
        x, y = float(t[i, 0]), float(t[i, 1])
        c = tuple(colors[i].tolist())
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=c, outline=c)

    return pil


# ---------------------------------------------------------------------------
# Combined builder: produces all tracking visualizations for a single step
# ---------------------------------------------------------------------------

def build_tracking_visualizations(
    source_image: torch.Tensor,
    target_image: torch.Tensor,
    tracking_result: dict,
    model_output: dict,
    matching_pipeline,
    config,
    batch_idx: int = 0,
) -> Dict[str, Image.Image]:
    """
    Generate all tracking visualizations for WandB logging.

    Args:
        source_image:      [B, 3, H, W]
        target_image:      [B, 3, H, W]
        tracking_result:   output of ``compute_tracking_losses``.
        model_output:      dict from MatcherModel.forward().
        matching_pipeline: Matcher instance.
        config:            DotMap config.
        batch_idx:         which batch element to visualize.

    Returns:
        dict mapping visualization name -> PIL Image.
    """
    from matching import correspondence
    from utilities.tensor_utils import embedding2chw

    fwd = tracking_result["fwd_result"]
    bwd = tracking_result["bwd_result"]
    query_pts = tracking_result["query_points"]
    tracked_pts = fwd["tracked_points"]
    coarse_tgt = fwd["coarse_target"]
    scores = fwd["scores"]
    roundtrip = bwd["tracked_points"]
    vis_logit = fwd.get("visibility_logit")

    images = {}

    # D1: Track lines
    images["track_lines"] = viz_track_lines(
        source_image, target_image, query_pts, tracked_pts, scores,
        batch_idx=batch_idx,
    )

    # D2: Coarse vs fine
    images["coarse_vs_fine"] = viz_coarse_vs_fine(
        target_image, coarse_tgt, tracked_pts, scores,
        batch_idx=batch_idx,
    )

    # D3: Cycle consistency error
    images["cycle_error"] = viz_cycle_error(
        source_image, query_pts, roundtrip,
        batch_idx=batch_idx,
    )

    # D4: Gated layer heatmap
    gated_maps = getattr(matching_pipeline.model, "latest_gated_layer_maps", {})
    src_gated = gated_maps.get("source") if gated_maps else None
    diagnostics = getattr(matching_pipeline.model, "latest_diagnostics", {})
    src_diag = diagnostics.get("source") if diagnostics else None

    if src_gated is not None and src_diag is not None:
        layer_indices = src_diag.get("layer_indices")
        if layer_indices is not None:
            images["gated_layer_heatmap"] = viz_gated_layer_heatmap(
                source_image, query_pts, src_gated, layer_indices,
                matching_pipeline.patch_size, batch_idx=batch_idx,
            )

    # D5: Visibility
    if vis_logit is not None:
        images["visibility"] = viz_visibility(
            target_image, tracked_pts, vis_logit,
            batch_idx=batch_idx,
        )

    # D6: Descriptor similarity
    patch_size = matching_pipeline.patch_size
    with torch.no_grad():
        src_map = embedding2chw(model_output["source_embedding_match"], embed_dim_last=False)
        tgt_map = embedding2chw(model_output["target_embedding_match"], embed_dim_last=False)
        src_emb = correspondence.sample_embeddings_at_points(src_map, query_pts, patch_size)
        tgt_emb = correspondence.sample_embeddings_at_points(tgt_map, tracked_pts, patch_size)

    images["descriptor_similarity"] = viz_descriptor_similarity(
        target_image, tracked_pts, src_emb, tgt_emb,
        batch_idx=batch_idx,
    )

    return images
