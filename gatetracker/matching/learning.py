import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_grid(
    num_points,
    batch_size=1,
    image_height=None,
    image_width=None,
    framestack=None,
    device="cuda",
):
    """
    Generate random matching points within image dimensions.

    Args:
        num_points (int): Number of points to generate
        batch_size (int): Batch size for generated points
        image_height (int, optional): Height of the image. Inferred from framestack if not provided
        image_width (int, optional): Width of the image. Inferred from framestack if not provided
        framestack (torch.Tensor, optional): Input framestack of shape [B, T, C, H, W]
        device (str): Device to place tensors on

    Returns:
        torch.Tensor: Random points of shape [batch_size, num_points, 2]
    """
    if framestack is not None:
        image_height = framestack.shape[-2]
        image_width = framestack.shape[-1]
        batch_size = framestack.shape[0]

    assert (
        image_height is not None and image_width is not None
    ), "Must provide either framestack or image dimensions"

    x_coords = torch.rand(batch_size, num_points) * (image_width - 1)
    y_coords = torch.rand(batch_size, num_points) * (image_height - 1)

    points = torch.stack([x_coords, y_coords], dim=-1).to(device)

    return points


def mine_triplets_optimized(
    sourceembs,
    targetembs,
    margin=0.0,
    nth_candidate=1,
    target_mask=None,
    match_confidence=None,
    sourceembs_idx=None,
    targetembs_idx=None,
    are_matched=True,
):
    """
    Vectorized version of the 'triplets' function, returning the same
    triplets (possibly in a different order) but without explicit for-loops
    over batch or anchor indices.
    """

    B, C, S = sourceembs.shape

    src_norm = F.normalize(sourceembs, p=2, dim=1)  # (B, C, S)
    tgt_norm = F.normalize(targetembs, p=2, dim=1)  # (B, C, S)

    sim_src2tgt = torch.matmul(src_norm.transpose(1, 2), tgt_norm)  # (B, S, S)
    dist_src2tgt = torch.sqrt(torch.clamp(2.0 - 2.0 * sim_src2tgt, min=0.0))

    sim_tgt2tgt = torch.matmul(tgt_norm.transpose(1, 2), tgt_norm)  # (B, S, S)
    dist_tgt2tgt = torch.sqrt(torch.clamp(2.0 - 2.0 * sim_tgt2tgt, min=0.0))

    sim_matrix = sim_src2tgt

    if target_mask is not None:
        mask_reduced = target_mask[:, 0, :].int()  # (B, S)
    else:
        mask_reduced = None

    if mask_reduced is None:
        valid_targets = torch.ones((B, S), dtype=torch.bool, device=sourceembs.device)
    else:
        valid_targets = mask_reduced.bool()  # (B, S)

    anchor_grid = torch.arange(S, device=sourceembs.device).unsqueeze(0).expand(B, S)

    if are_matched:
        pos_idx = anchor_grid
        valid_pos_mask = valid_targets.gather(dim=1, index=pos_idx)
    else:
        dist_for_pos = dist_src2tgt  # shape (B, S, S)
        pos_condition = (dist_for_pos < margin).logical_and(
            valid_targets.unsqueeze(1)  # shape (B,1,S)
        )

        dist_for_pos_valid = torch.where(
            pos_condition, dist_for_pos, torch.full_like(dist_for_pos, float("inf"))
        )
        values, indices = dist_for_pos_valid.topk(k=nth_candidate, dim=2, largest=False)
        nth_vals = values[:, :, nth_candidate - 1]  # (B, S)
        pos_idx = indices[:, :, nth_candidate - 1]  # (B, S)
        valid_pos_mask = torch.isfinite(nth_vals)

    dist_for_negA = dist_src2tgt  # (B, S, S)
    negA_condition = (dist_for_negA > margin).logical_and(
        valid_targets.unsqueeze(1)  # shape (B,1,S)
    )

    j_grid = torch.arange(S, device=sourceembs.device).view(1, 1, S)
    pos_idx_3d = pos_idx.unsqueeze(-1)  # (B, S, 1)
    mask_j_not_pos = j_grid.expand(B, S, S) != pos_idx_3d

    negA_condition = negA_condition & mask_j_not_pos

    dist_for_negA_valid = torch.where(
        negA_condition, dist_for_negA, torch.full_like(dist_for_negA, float("inf"))
    )
    valsA, idxA = dist_for_negA_valid.topk(k=nth_candidate, dim=2, largest=False)
    negA_idx = idxA[:, :, nth_candidate - 1]  # (B, S)
    negA_dist = valsA[:, :, nth_candidate - 1]  # (B, S)
    valid_negA_mask = torch.isfinite(negA_dist)

    row_indices = pos_idx.clamp(min=0, max=S - 1)
    b_grid = torch.arange(B, device=sourceembs.device).view(B, 1, 1).expand(B, S, S)
    i_grid = torch.arange(S, device=sourceembs.device).view(1, S, 1).expand(B, S, S)
    j_grid = torch.arange(S, device=sourceembs.device).view(1, 1, S).expand(B, S, S)
    row_gather = row_indices.unsqueeze(-1).expand(B, S, S)  # shape (B,S,S)
    dist_for_negB = dist_tgt2tgt[b_grid, row_gather, j_grid]  # (B, S, S)

    negB_condition = (dist_for_negB > margin).logical_and(valid_targets.unsqueeze(1))
    mask_j_not_posB = j_grid != row_gather
    negB_condition = negB_condition & mask_j_not_posB

    dist_for_negB_valid = torch.where(
        negB_condition, dist_for_negB, torch.full_like(dist_for_negB, float("inf"))
    )
    valsB, idxB = dist_for_negB_valid.topk(k=nth_candidate, dim=2, largest=False)
    negB_idx = idxB[:, :, nth_candidate - 1]  # (B, S)
    negB_dist = valsB[:, :, nth_candidate - 1]  # (B, S)
    valid_negB_mask = torch.isfinite(negB_dist)

    both_valid_mask = valid_negA_mask & valid_negB_mask
    onlyA_valid_mask = valid_negA_mask & ~valid_negB_mask
    onlyB_valid_mask = ~valid_negA_mask & valid_negB_mask

    pickA_mask = both_valid_mask & (negA_dist <= negB_dist)
    pickB_mask = both_valid_mask & (negB_dist < negA_dist)

    final_neg_idx = torch.where(
        pickA_mask,
        negA_idx,
        torch.where(
            pickB_mask,
            negB_idx,
            torch.where(
                onlyA_valid_mask,
                negA_idx,
                torch.where(onlyB_valid_mask, negB_idx, torch.full_like(negA_idx, -1)),
            ),
        ),
    )

    final_neg_dist = torch.where(
        pickA_mask,
        negA_dist,
        torch.where(
            pickB_mask,
            negB_dist,
            torch.where(
                onlyA_valid_mask,
                negA_dist,
                torch.where(
                    onlyB_valid_mask,
                    negB_dist,
                    torch.full_like(negA_dist, float("inf")),
                ),
            ),
        ),
    )

    valid_trip_mask = (
        valid_pos_mask & (final_neg_idx >= 0) & torch.isfinite(final_neg_dist)
    )

    lin_bi = torch.arange(B * S, device=sourceembs.device)
    b_idx = lin_bi // S
    i_idx = lin_bi % S

    valid_trip_mask_flat = valid_trip_mask.view(-1)
    final_neg_idx_flat = final_neg_idx.view(-1)
    pos_idx_flat = pos_idx.reshape(-1)

    valid_lin_bi = lin_bi[valid_trip_mask_flat]
    valid_b = b_idx[valid_trip_mask_flat]
    valid_i = i_idx[valid_trip_mask_flat]
    valid_pos = pos_idx_flat[valid_trip_mask_flat]
    valid_neg = final_neg_idx_flat[valid_trip_mask_flat]
    a2p_dist = dist_src2tgt.view(-1, S)[valid_lin_bi, valid_pos]  # shape (#triplets,)
    a2n_dist = final_neg_dist.view(-1)[valid_trip_mask_flat]

    anchor_emb = sourceembs[valid_b, :, valid_i]
    positive_emb = targetembs[valid_b, :, valid_pos]
    negative_emb = targetembs[valid_b, :, valid_neg]

    if sourceembs_idx is not None:
        anchor_id = sourceembs_idx[valid_b, valid_i]
    else:
        anchor_id = valid_i
    if targetembs_idx is not None:
        positive_id = targetembs_idx[valid_b, valid_pos]
        negative_id = targetembs_idx[valid_b, valid_neg]
    else:
        positive_id = valid_pos
        negative_id = valid_neg

    if match_confidence is None:
        match_confidence = torch.ones((B, S), device=sourceembs.device)
    elif match_confidence.ndim == 3:
        match_confidence = match_confidence[:, 0, :]
    match_confidence = match_confidence.float()
    triplet_confidence = match_confidence[valid_b, valid_i]

    valid_target_counts = valid_targets.sum(dim=1, keepdim=True)  # [B, 1]
    contrastive_row_mask = valid_targets & (valid_target_counts > 1)  # [B, S]
    contrastive_row_mask_flat = contrastive_row_mask.reshape(-1)
    contrastive_b = b_idx[contrastive_row_mask_flat]
    contrastive_i = i_idx[contrastive_row_mask_flat]

    if contrastive_b.numel() == 0:
        source_to_target_logits = sim_src2tgt.reshape(-1, S)[:0]
        target_to_source_logits = sim_src2tgt.transpose(1, 2).reshape(-1, S)[:0]
        contrastive_mask = torch.zeros((0, S), dtype=torch.bool, device=sourceembs.device)
        contrastive_positive = torch.zeros((0,), dtype=torch.long, device=sourceembs.device)
        contrastive_weights = match_confidence.reshape(-1)[:0]
        contrastive_batch = torch.zeros((0,), dtype=torch.long, device=sourceembs.device)
        contrastive_anchor_idx = torch.zeros((0,), dtype=torch.long, device=sourceembs.device)
    else:
        source_to_target_logits = sim_src2tgt[contrastive_b, contrastive_i]  # [M, S]
        target_to_source_logits = sim_src2tgt.transpose(1, 2)[contrastive_b, contrastive_i]  # [M, S]
        contrastive_mask = valid_targets[contrastive_b]  # [M, S]
        contrastive_positive = contrastive_i.long()  # [M]
        contrastive_weights = match_confidence[contrastive_b, contrastive_i].clamp_(0.0, 1.0)  # [M]
        contrastive_batch = contrastive_b  # [M]
        if sourceembs_idx is not None:
            contrastive_anchor_idx = sourceembs_idx[contrastive_b, contrastive_i]
        else:
            contrastive_anchor_idx = contrastive_i

    out = {
        "anchor": anchor_emb,
        "positive": positive_emb,
        "negative": negative_emb,
        "a2p": a2p_dist,
        "a2n": a2n_dist,
        "sim_matrix": sim_matrix,  # shape (B, S, S), same as original
        "anchor_indices": anchor_id,
        "positive_indices": positive_id,
        "negative_indices": negative_id,
        "batch_indices": valid_b,
        "triplet_confidence": triplet_confidence,
        "source_to_target_logits": source_to_target_logits,
        "source_to_target_mask": contrastive_mask,
        "source_to_target_positive": contrastive_positive,
        "target_to_source_logits": target_to_source_logits,
        "target_to_source_mask": torch.ones_like(contrastive_mask, dtype=torch.bool),
        "target_to_source_positive": contrastive_positive,
        "contrastive_weights": contrastive_weights,
        "contrastive_batch_indices": contrastive_batch,
        "contrastive_anchor_indices": contrastive_anchor_idx,
    }
    return out
