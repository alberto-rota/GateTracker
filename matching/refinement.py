from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def process_patch(patch_data):
    s_patch, t_patch, ratio_thresh, patch_size = patch_data

    # Convert source patch: (3,H,W) → (H,W,3), scale to uint8, grayscale
    s_patch = np.transpose(s_patch, (2, 1, 0))
    s_patch_8u = (s_patch * 255).clip(0, 255).astype(np.uint8)
    s_patch_gray = cv2.cvtColor(s_patch_8u, cv2.COLOR_RGB2GRAY)

    # Convert target patch similarly
    t_patch = np.transpose(t_patch, (2, 1, 0))
    t_patch_8u = (t_patch * 255).clip(0, 255).astype(np.uint8)
    t_patch_gray = cv2.cvtColor(t_patch_8u, cv2.COLOR_RGB2GRAY)

    # Create SIFT and BF matcher (using CPU)
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2)
    kp_s, desc_s = sift.detectAndCompute(s_patch_gray, None)
    kp_t, desc_t = sift.detectAndCompute(t_patch_gray, None)

    # Fallback if no keypoints/descriptors found
    if desc_s is None or desc_t is None or len(desc_s) == 0 or len(desc_t) == 0:
        return (
            [patch_size / 2, patch_size / 2],
            [patch_size / 2, patch_size / 2],
            1 - ratio_thresh,
        )

    # kNN matching and Lowe's ratio test
    knn_matches = bf.knnMatch(desc_s, desc_t, k=2)
    good_matches = []
    good_scores = []
    for match_pair in knn_matches:
        if len(match_pair) >= 2:
            m, n = match_pair
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
                good_scores.append(
                    1 - (1 - ratio_thresh) / ratio_thresh * (m.distance / n.distance)
                )

    if len(good_matches) == 0:
        return (
            [patch_size / 2, patch_size / 2],
            [patch_size / 2, patch_size / 2],
            1 - ratio_thresh,
        )

    # Select best match (lowest distance)
    best_match = min(good_matches, key=lambda x: x.distance)
    best_score = min(good_scores)
    s_kp = kp_s[best_match.queryIdx].pt  # (x, y)
    t_kp = kp_t[best_match.trainIdx].pt  # (x, y)

    return ([s_kp[1], s_kp[0]], [t_kp[1], t_kp[0]], best_score)


def SIFT_patch_refiner(
    spatches: torch.Tensor,
    tpatches: torch.Tensor,
    ratio_thresh=0.75,
    patch_size=16,
    num_workers=8,
):
    N = spatches.shape[0]
    # Convert all patches to numpy arrays once (CPU-side)
    spatches_np = spatches.cpu().numpy()
    tpatches_np = tpatches.cpu().numpy()
    # Prepare data for parallel processing
    patch_data = [
        (spatches_np[i], tpatches_np[i], ratio_thresh, patch_size) for i in range(N)
    ]
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_patch, patch_data))
    src_coords_list, tgt_coords_list, scores_list = zip(*results)
    device = spatches.device
    best_src_xy = torch.tensor(src_coords_list, dtype=torch.float, device=device)
    best_tgt_xy = torch.tensor(tgt_coords_list, dtype=torch.float, device=device)
    best_scores = torch.tensor(scores_list, dtype=torch.float, device=device)
    return best_src_xy, best_tgt_xy, best_scores


def FFT_patch_refiner(
    spatches: torch.Tensor,
    tpatches: torch.Tensor,
    patch_size: int = 16,
    confidence_threshold: float = None,
):
    """
    Refines patch correspondences using phase correlation with torch tensors on GPU.
    Incorporates original matching scores and produces refined scores based on correlation confidence.

    Inputs:
      spatches: Source patches tensor of shape (N, C, patch_size, patch_size)
      tpatches: Target patches tensor of shape (N, C, patch_size, patch_size)
      scores: Original confidence scores for each match of shape (N,)
      confidence_threshold: Minimum correlation value to consider a refinement valid
      patch_size: The size of each square patch
      alpha: Weight for blending original and correlation scores (higher alpha gives more weight to original scores)

    Outputs:
      best_src_xy: Tensor of refined source patch centers (N, 2)
      best_tgt_xy: Tensor of corresponding target patch centers (N, 2)
      refined_scores: Tensor of refined confidence scores (N,)
    """
    N = spatches.shape[0]
    eps = 1e-8

    # Compute FFT for each patch pair (batch-wise)
    F1 = torch.fft.fft2(spatches, dim=(-2, -1))
    F2 = torch.fft.fft2(tpatches, dim=(-2, -1))

    # Compute cross-power spectrum and sum over channels
    cross_power = (F2 * torch.conj(F1)).sum(dim=1, keepdim=True)
    cross_power_normalized = cross_power / (torch.abs(cross_power) + eps)

    # Compute the inverse FFT to obtain the correlation surface
    corr = torch.fft.ifft2(cross_power_normalized, dim=(-2, -1))
    corr_abs = torch.abs(corr).squeeze(1)  # Shape: (N, patch_size, patch_size)

    # Find the peak correlation value and its index for each patch
    best_values, best_indices = corr_abs.view(N, -1).max(dim=1)  # Shapes: (N,), (N,)

    # Convert flat indices into 2D indices (peak_y, peak_x)
    peak_y = best_indices // patch_size
    peak_x = best_indices % patch_size

    # Compute offsets, adjusting for wrap-around
    offset_x = torch.where(peak_x > (patch_size // 2), peak_x - patch_size, peak_x)
    offset_y = torch.where(peak_y > (patch_size // 2), peak_y - patch_size, peak_y)
    offsets = torch.stack((offset_x, offset_y), dim=1).float()  # shape: (N, 2)

    # Define the center of a patch
    center = (patch_size - 1) / 2.0
    best_src_xy = torch.tensor([center, center], device=spatches.device).repeat(N, 1)
    best_tgt_xy = best_src_xy + offsets

    # Create refined scores by blending original scores with correlation values
    # Normalize correlation values to [0, 1] range (they should already be in this range)
    normalized_corr = best_values.clamp(0, 1)
    # Create a mask for matches with correlation below threshold
    # low_confidence_mask = normalized_corr > confidence_threshold
    # # Blend scores using weighted average for matches with sufficient confidence
    # refined_scores = torch.where(
    #     low_confidence_mask,
    #     # torch.zeros_like(scores),  # Set score to 0 for low confidence refinements
    #     alpha * scores + (1 - alpha) * normalized_corr,  # Weighted blend for others
    #     normalized_corr,
    # )

    if confidence_threshold is not None:
        keep_mask = normalized_corr >= float(confidence_threshold)
        best_tgt_xy = torch.where(keep_mask.unsqueeze(1), best_tgt_xy, best_src_xy)
        normalized_corr = torch.where(
            keep_mask, normalized_corr, torch.zeros_like(normalized_corr)
        )

    return best_src_xy, best_tgt_xy, normalized_corr


def feature_refinement_patch_size(window_radius: int, feature_stride: int) -> int:
    """
    Returns the virtual patch size used by `apply_refinement_offsets`.

    Args:
        window_radius: Search radius in fine-feature coordinates.
        feature_stride: Pixel stride between image and fine feature map.
    """
    return 2 * int(window_radius) * int(feature_stride) + 1


def _pixels_to_feature_coords(points: torch.Tensor, feature_stride: int) -> torch.Tensor:
    """
    Maps image-space pixel centers to fine-feature coordinates.

    Args:
        points: [M, 2] pixel coordinates in `(x, y)` order.
        feature_stride: Pixel stride of the fine feature map.

    Returns:
        [M, 2] coordinates in feature-map index space.
    """
    return (points + 0.5) / float(feature_stride) - 0.5


def _coords_to_grid(coords: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    Converts feature coordinates to `grid_sample` coordinates.

    Args:
        coords: [..., 2] feature coordinates in `(x, y)` order.
        height: Feature-map height.
        width: Feature-map width.

    Returns:
        [..., 2] normalized coordinates in `[-1, 1]`.
    """
    grid = coords.clone()
    grid[..., 0] = (grid[..., 0] / max(width - 1, 1)) * 2 - 1
    grid[..., 1] = (grid[..., 1] / max(height - 1, 1)) * 2 - 1
    return grid


def feature_softargmax_refiner(
    source_feature_map: torch.Tensor,
    target_feature_map: torch.Tensor,
    source_pixels: torch.Tensor,
    target_pixels: torch.Tensor,
    batch_indices: torch.Tensor,
    window_radius: int = 2,
    feature_stride: int = 4,
    softmax_temperature: float = 0.1,
    confidence_threshold: float = None,
):
    """
    Refines coarse correspondences in feature space with local correlation.

    Args:
        source_feature_map: [B, C_f, H_f, W_f] fine source features.
        target_feature_map: [B, C_f, H_f, W_f] fine target features.
        source_pixels: [M, 2] coarse source pixels in `(x, y)`.
        target_pixels: [M, 2] coarse target pixels in `(x, y)`.
        batch_indices: [M] batch index of each correspondence.
        window_radius: Local search radius in fine-feature cells.
        feature_stride: Pixel stride between image and fine feature map.
        softmax_temperature: Correlation temperature for soft-argmax.
        confidence_threshold: Optional threshold for gating low-confidence shifts.

    Returns:
        best_src_xy: [M, 2] source coordinates inside the virtual refinement patch.
        best_tgt_xy: [M, 2] target coordinates inside the virtual refinement patch.
        refinement_scores: [M] confidence scores in `[0, 1]`.
    """
    device = source_feature_map.device
    dtype = source_feature_map.dtype
    match_count = source_pixels.shape[0]
    patch_size = feature_refinement_patch_size(window_radius, feature_stride)
    patch_center = patch_size / 2.0 - 0.5
    best_src_xy = torch.full((match_count, 2), patch_center, device=device, dtype=dtype)
    if match_count == 0:
        return best_src_xy, best_src_xy.clone(), torch.zeros(0, device=device, dtype=dtype)

    _, _, feature_height, feature_width = source_feature_map.shape
    source_maps = source_feature_map[batch_indices]  # [M, C_f, H_f, W_f]
    target_maps = target_feature_map[batch_indices]  # [M, C_f, H_f, W_f]

    source_feature_coords = _pixels_to_feature_coords(source_pixels.float(), feature_stride)
    target_feature_coords = _pixels_to_feature_coords(target_pixels.float(), feature_stride)

    source_grid = _coords_to_grid(
        source_feature_coords.view(match_count, 1, 1, 2),
        feature_height,
        feature_width,
    )  # [M, 1, 1, 2]
    source_queries = F.grid_sample(
        source_maps,
        source_grid,
        mode="bilinear",
        align_corners=True,
    ).squeeze(-1).squeeze(-1)  # [M, C_f]
    source_queries = F.normalize(source_queries, dim=1)

    offsets = torch.arange(
        -window_radius, window_radius + 1, device=device, dtype=dtype
    )  # [W_w]
    offset_y, offset_x = torch.meshgrid(offsets, offsets, indexing="ij")
    offset_grid = torch.stack((offset_x, offset_y), dim=-1)  # [W_w, W_w, 2]
    target_window_coords = (
        target_feature_coords.view(match_count, 1, 1, 2) + offset_grid.unsqueeze(0)
    )  # [M, W_w, W_w, 2]
    target_grid = _coords_to_grid(target_window_coords, feature_height, feature_width)
    target_windows = F.grid_sample(
        target_maps,
        target_grid,
        mode="bilinear",
        align_corners=True,
    )  # [M, C_f, W_w, W_w]
    target_windows = F.normalize(target_windows, dim=1)

    logits = torch.einsum(
        "mc,mcij->mij",
        source_queries,
        target_windows,
    )  # [M, W_w, W_w]
    logits = logits.view(match_count, -1) / max(float(softmax_temperature), 1e-6)
    probabilities = torch.softmax(logits, dim=1)  # [M, W_w * W_w]
    window_width = 2 * window_radius + 1
    probabilities_2d = probabilities.view(match_count, window_width, window_width)

    offset_grid_pixels = offset_grid.view(1, window_width, window_width, 2) * float(
        feature_stride
    )  # [1, W_w, W_w, 2]
    expected_offsets = (
        probabilities_2d.unsqueeze(-1) * offset_grid_pixels
    ).sum(dim=(1, 2))  # [M, 2]
    best_tgt_xy = best_src_xy + expected_offsets

    peak_probability = probabilities.max(dim=1).values  # [M]
    entropy = -(probabilities * torch.log(probabilities + 1e-8)).sum(dim=1)  # [M]
    max_entropy = torch.log(
        torch.tensor(float(probabilities.shape[1]), device=device, dtype=dtype)
    ).clamp_min(1e-6)
    entropy_confidence = (1.0 - entropy / max_entropy).clamp(0.0, 1.0)
    refinement_scores = (0.5 * peak_probability + 0.5 * entropy_confidence).clamp(
        0.0, 1.0
    )  # [M]

    if confidence_threshold is not None:
        keep_mask = refinement_scores >= float(confidence_threshold)
        best_tgt_xy = torch.where(keep_mask.unsqueeze(1), best_tgt_xy, best_src_xy)
        refinement_scores = torch.where(
            keep_mask, refinement_scores, torch.zeros_like(refinement_scores)
        )

    return best_src_xy, best_tgt_xy, refinement_scores
