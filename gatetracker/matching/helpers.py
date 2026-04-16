import torch
import torch.nn.functional as F
import numpy as np


def points_to_patches(points, embedding_map, patch_size=8, mask=None):
    B, N, _ = points.shape
    _, C, H_p, W_p = embedding_map.shape
    H, W = H_p * patch_size, W_p * patch_size

    points = torch.clamp(points, min=0, max=float(H))
    patch_x = torch.clamp((points[..., 0] / patch_size).long(), 0, W_p - 1)
    patch_y = torch.clamp((points[..., 1] / patch_size).long(), 0, H_p - 1)

    batch_idx = torch.arange(B, device=points.device)[:, None, None]
    channel_idx = torch.arange(C, device=points.device)[None, :, None]
    patch_y_exp = patch_y.unsqueeze(1).expand(B, C, N)
    patch_x_exp = patch_x.unsqueeze(1).expand(B, C, N)

    output = embedding_map[batch_idx, channel_idx, patch_y_exp, patch_x_exp]

    if mask is not None:
        mask_reduced = mask[:, 0]
        output_mask = mask_reduced[
            torch.arange(B, device=points.device)[:, None], patch_y, patch_x
        ]
        output_mask = output_mask.unsqueeze(1).expand(B, C, N)
    else:
        output_mask = torch.ones((B, C, N), device=points.device)

    flat_indices = patch_y * W_p + patch_x
    return output, flat_indices, output_mask


def filter_inliers(points_data, inliers_mask):
    """
    Filter all point-related data based on inlier mask.

    Args:
        points_data: Dictionary containing all point-related data
        inliers_mask: Boolean mask indicating inliers

    Returns:
        Dictionary with filtered point data
    """
    filtered_data = {}
    for key, value in points_data.items():
        if value is not None:
            filtered_data[key] = value[inliers_mask]

    return filtered_data


def filter_scores(points_data, scores, threshold=0):
    """
    Filter all point-related data based on score threshold.

    Args:
        points_data: Dictionary containing all point-related data
        scores: Confidence scores for matches
        threshold: Minimum score to keep

    Returns:
        Dictionary with filtered point data and valid scores
    """
    valid_mask = scores != threshold
    filtered_data = filter_inliers(points_data, valid_mask)
    filtered_scores = scores[valid_mask]

    return filtered_data, filtered_scores


def apply_refinement_offsets(
    source_pixels, target_pixels, source_offsets, target_offsets, patch_size
):
    """
    Apply refinement offsets to the matched pixel coordinates.

    Args:
        source_pixels: Source frame pixel coordinates
        target_pixels: Target frame pixel coordinates
        source_offsets: Offset values for source pixels
        target_offsets: Offset values for target pixels
        patch_size: Size of the patch used for refinement

    Returns:
        Updated source and target pixel coordinates
    """
    half_patch = patch_size / 2

    source_pixels_refined = source_pixels + source_offsets - half_patch + 0.5
    target_pixels_refined = target_pixels + target_offsets - half_patch + 0.5

    return source_pixels_refined, target_pixels_refined
