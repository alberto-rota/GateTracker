import numpy as np
import torch


def inlier_ratio(pred_points, true_points, threshold):
    distances = torch.norm(pred_points - true_points, dim=2)
    correct_matches = distances < threshold

    inlier_ratios = correct_matches.sum(dim=1) / correct_matches.size(1)
    return inlier_ratios.mean().item()  # Average across batche


def descriptor_matching_rate(pred_points, true_points, threshold):
    distances = torch.norm(pred_points - true_points, dim=2)
    correct_matches = distances < threshold

    matching_rate = correct_matches.sum(dim=1) / pred_points.size(1)
    return matching_rate.mean().item()


def NCM(pred_points, true_points, threshold):
    """
    Calculates the Number of Correct Matches (NCM).

    Args:
        pred_points (torch.Tensor): Predicted points of shape (B, N, 2).
        true_points (torch.Tensor): Ground truth points of shape (B, N, 2).
        threshold (float): Distance threshold for considering a match correct.

    Returns:
        int: Total number of correct matches across all batches.
    """
    distances = torch.norm(pred_points - true_points, dim=2)  # Euclidean distance
    correct_matches = (distances < threshold).sum()  # Count correct matches
    return correct_matches.item()  # Return as integer


def success_rate(pred_points, true_points, threshold, success_threshold=0.5):
    """
    Calculates the Success Rate (SR).

    Args:
        pred_points (torch.Tensor): Predicted points of shape (B, N, 2).
        true_points (torch.Tensor): Ground truth points of shape (B, N, 2).
        threshold (float): Distance threshold for considering a match correct.
        success_threshold (float): Proportion of correct matches required to consider a batch successful (default=0.5).

    Returns:
        float: Success rate as a fraction of successful batches.
    """
    distances = torch.norm(pred_points - true_points, dim=2)  # Euclidean distance
    correct_matches_per_batch = (distances < threshold).sum(
        dim=1
    )  # Count correct matches per batch
    match_ratios = correct_matches_per_batch / pred_points.size(
        1
    )  # Proportion of correct matches per batch

    successful_batches = (
        match_ratios >= success_threshold
    ).sum()  # Count successful batches
    success_rate = successful_batches / pred_points.size(
        0
    )  # Fraction of successful batches
    return success_rate.item()  # Return as a float


def fundamental_error(F1, F2, error_type="percentage", reduction="mean"):
    """
    Calculate the error between two batches of fundamental matrices.
    Handles sign ambiguity, scale differences, and batching.

    Args:
        F1 (torch.Tensor): First batch of 3x3 fundamental matrices (shape: Bx3x3)
        F2 (torch.Tensor): Second batch of 3x3 fundamental matrices (shape: Bx3x3)
        error_type (str): Type of error to return ('percentage' or 'absolute')
        reduction (str): Reduction method ('mean' or 'none')

    Returns:
        If reduction='mean':
            float: Average error across the batch (percentage or absolute)
            torch.Tensor: Individual errors for each matrix in the batch
        If reduction='none':
            torch.Tensor: Individual errors for each matrix in the batch
    """
    if F1.ndim == 2:
        F1 = F1.unsqueeze(0)
    elif F1.ndim == 4:
        F1 = F1.squeeze()
        if F1.ndim == 2:
            F1 = F1.unsqueeze(0)
        elif F1.ndim == 4:
            if F1.shape[1] == 1:
                F1 = F1.squeeze(1)

    if F2.ndim == 2:
        F2 = F2.unsqueeze(0)
    elif F2.ndim == 4:
        F2 = F2.squeeze()
        if F2.ndim == 2:
            F2 = F2.unsqueeze(0)
        elif F2.ndim == 4:
            if F2.shape[1] == 1:
                F2 = F2.squeeze(1)

    if F1.ndim != 3 or F2.ndim != 3:
        raise ValueError(f"Input shapes must be normalized to Bx3x3. Got {F1.shape} and {F2.shape}")
    if F1.shape != F2.shape:
        raise ValueError(f"Input shapes must match. Got {F1.shape} and {F2.shape}")

    batch_size = F1.shape[0]

    F1_norms = torch.norm(F1.reshape(batch_size, -1), dim=1) + 1e-6
    F2_norms = torch.norm(F2.reshape(batch_size, -1), dim=1) + 1e-6

    F1_norm = F1 / F1_norms.view(-1, 1, 1)
    F2_norm = F2 / F2_norms.view(-1, 1, 1)

    error_positive = torch.norm((F1_norm - F2_norm).reshape(batch_size, -1), dim=1)
    error_negative = torch.norm((F1_norm + F2_norm).reshape(batch_size, -1), dim=1)

    batch_errors = torch.minimum(error_positive, error_negative)

    if error_type == "percentage":
        max_possible_error = torch.sqrt(torch.tensor(2.0, device=F1.device))
        batch_errors = batch_errors / max_possible_error

    if torch.any(F1_norms <= 5e-6) or torch.any(F2_norms <= 5e-6):
        batch_errors = torch.full((batch_size,), float("nan"), device=F1.device)
        avg_error = float("nan")
    else:
        avg_error = torch.mean(batch_errors).item() if reduction == "mean" else None

    if reduction == "mean":
        return avg_error, batch_errors
    else:
        return batch_errors


def epipolar_error(points1, points2, F, batch_index, reduction="mean"):
    """
    Calculate epipolar error between matched points using fundamental matrices.

    Args:
        points1 (torch.Tensor): First set of points (shape: Nx2)
        points2 (torch.Tensor): Second set of points (shape: Nx2)
        F (torch.Tensor): Batch of fundamental matrices (shape: Bx3x3)
        batch_index (torch.Tensor): Batch indices for each point pair (shape: Nx1)
        reduction (str): Reduction method ('mean' or 'none')

    Returns:
        If reduction='mean':
            float: Mean epipolar error
        If reduction='none':
            torch.Tensor: Per-point epipolar errors (shape: N)
    """
    N = points1.shape[0]
    ones = torch.ones(N, 1, device=points1.device, dtype=points1.dtype)
    p1_h = torch.cat([points1, ones], dim=1)  # shape Nx3
    p2_h = torch.cat([points2, ones], dim=1)  # shape Nx3

    if F.ndim == 2:
        F = F.unsqueeze(0)
    elif F.ndim == 4:
        F = F.squeeze()
        if F.ndim == 2:
            F = F.unsqueeze(0)

    device = points1.device
    F = F.to(device)

    batch_idx = batch_index.squeeze(-1).int()  # shape N
    if F.shape[0] == 1:
        zero_idx = torch.zeros(N, dtype=torch.long, device=device)
        F_sel = F[zero_idx]  # shape Nx3x3
    else:
        F_sel = F[batch_idx]  # shape Nx3x3

    F_sel = F_sel.squeeze()
    if F_sel.ndim == 2:
        F_sel = F_sel.unsqueeze(0)
    elif F_sel.ndim == 4:
        F_sel = F_sel.squeeze(1)  # shape Nx3x3

    if F_sel.ndim != 3:
        raise ValueError(f"F_sel must be 3D (Nx3x3) for bmm. Got shape {F_sel.shape}")
    F_sel = F_sel.contiguous()

    p1_h_exp = p1_h.unsqueeze(-1).float()  # shape Nx3x1
    p2_h_exp = p2_h.unsqueeze(-1).float()  # shape Nx3x1

    if p1_h_exp.ndim != 3 or p2_h_exp.ndim != 3:
        raise ValueError(f"Point tensors must be 3D for bmm. Got shapes {p1_h_exp.shape}, {p2_h_exp.shape}")
    p1_h_exp = p1_h_exp.contiguous()
    p2_h_exp = p2_h_exp.contiguous()

    l2 = torch.bmm(F_sel, p1_h_exp).squeeze(-1)  # shape Nx3
    l1 = torch.bmm(F_sel.transpose(1, 2), p2_h_exp).squeeze(-1)  # shape Nx3

    a2, b2, c2 = l2.unbind(dim=1)
    a1, b1, c1 = l1.unbind(dim=1)
    x2, y2 = points2.unbind(dim=1)
    x1, y1 = points1.unbind(dim=1)

    dist2 = torch.abs(a2 * x2 + b2 * y2 + c2) / torch.sqrt(a2**2 + b2**2 + 1e-8)
    dist1 = torch.abs(a1 * x1 + b1 * y1 + c1) / torch.sqrt(a1**2 + b1**2 + 1e-8)
    error = (dist1 + dist2) / 2.0

    if reduction == "mean":
        return error.mean().item()
    else:
        return error


def mean_matching_distance(pred_points, true_points, batch_index, reduction="mean"):
    """
    Calculate mean distance between predicted and true points with batch support.

    Args:
        pred_points (torch.Tensor): Predicted points, shape (N, 2)
        true_points (torch.Tensor): Ground truth points, shape (N, 2)
        batch_index (torch.Tensor): Batch indices for each point pair, shape (N,) or (N, 1)
        reduction (str): Reduction method ('mean' or 'none')

    Returns:
        If reduction='mean':
            float: Mean distance across all batches
        If reduction='none':
            torch.Tensor: Mean distance for each batch, shape (B,) where B is the number of unique batches
    """
    if batch_index.dim() == 2:
        batch_index = batch_index.squeeze(-1)
    batch_index = batch_index.int()
    point_distances = torch.norm(pred_points.float() - true_points.float(), dim=1)

    if reduction == "mean":
        return point_distances.mean().item()
    else:
        unique_batches = torch.unique(batch_index)
        batch_distances = torch.zeros(len(unique_batches), device=pred_points.device)

        for i, batch_id in enumerate(unique_batches):
            batch_mask = batch_index == batch_id
            if batch_mask.sum() > 0:
                batch_distances[i] = point_distances[batch_mask].mean()

        return batch_distances


def refinement_metrics(
    mp,
    true_pixels_matched=None,
):
    """
    Computes refinement-specific diagnostics from the latest correspondence state.

    Args:
        mp: Matcher instance exposing `latest_refinement_state`.
        true_pixels_matched: [M, 2] pseudo-GT target pixels or None.

    Returns:
        dict: Refinement metrics suitable for console and W&B logging.
    """
    state = getattr(mp, "latest_refinement_state", None) or {}
    refined_scores = state.get("scores")
    coarse_target_pixels = state.get("coarse_target_pixels")
    refined_target_pixels = state.get("refined_target_pixels")
    active_mask = state.get("active_mask")

    empty_metrics = {
        "RefinementActiveFraction": None,
        "RefinementOffsetMean": None,
        "RefinementScoreMean": None,
        "CoarseErrorMean": None,
        "RefinedErrorMean": None,
        "RefinementGainPx": None,
        "RefinementGainRatio": None,
        "RefinementWinRate": None,
        "RefinementGainConfidenceCorr": None,
    }
    if (
        refined_scores is None
        or coarse_target_pixels is None
        or refined_target_pixels is None
        or refined_scores.numel() == 0
    ):
        return empty_metrics

    refined_scores = refined_scores.detach().float()  # [M]
    coarse_target_pixels = coarse_target_pixels.detach().float()  # [M, 2]
    refined_target_pixels = refined_target_pixels.detach().float()  # [M, 2]
    if active_mask is None:
        active_mask = torch.ones_like(refined_scores, dtype=torch.bool)
    else:
        active_mask = active_mask.detach().bool()  # [M]

    displacement = torch.norm(
        refined_target_pixels - coarse_target_pixels,
        dim=1,
    )  # [M]
    metrics_dict = {
        "RefinementActiveFraction": active_mask.float().mean().item(),
        "RefinementOffsetMean": displacement.mean().item(),
        "RefinementScoreMean": refined_scores.mean().item(),
        "CoarseErrorMean": None,
        "RefinedErrorMean": None,
        "RefinementGainPx": None,
        "RefinementGainRatio": None,
        "RefinementWinRate": None,
        "RefinementGainConfidenceCorr": None,
    }
    if true_pixels_matched is None or true_pixels_matched.numel() == 0:
        return metrics_dict

    true_pixels_matched = true_pixels_matched.detach().float()  # [M, 2]
    coarse_error = torch.norm(
        coarse_target_pixels - true_pixels_matched,
        dim=1,
    )  # [M]
    refined_error = torch.norm(
        refined_target_pixels - true_pixels_matched,
        dim=1,
    )  # [M]
    gain_px = coarse_error - refined_error  # [M]
    coarse_error_mean = coarse_error.mean()  # []
    refined_error_mean = refined_error.mean()  # []
    gain_px_mean = gain_px.mean()  # []

    metrics_dict.update(
        {
            "CoarseErrorMean": coarse_error_mean.item(),
            "RefinedErrorMean": refined_error_mean.item(),
            "RefinementGainPx": gain_px_mean.item(),
            "RefinementGainRatio": (
                gain_px_mean / coarse_error_mean.clamp_min(1e-6)
            ).item(),
            "RefinementWinRate": (gain_px > 0).float().mean().item(),
        }
    )

    if refined_scores.numel() > 1:
        score_std = refined_scores.std(unbiased=False)
        gain_std = gain_px.std(unbiased=False)
        if score_std > 1e-6 and gain_std > 1e-6:
            centered_scores = refined_scores - refined_scores.mean()
            centered_gain = gain_px - gain_px.mean()
            metrics_dict["RefinementGainConfidenceCorr"] = (
                (centered_scores * centered_gain).mean()
                / (score_std * gain_std + 1e-8)
            ).item()

    return metrics_dict


def precision_recall(
    source_pixels_matched,
    target_pixels_matched,
    true_pixels_matched,
    batch_indexes,
    confidence_scores,
    threshold,
    fundamental=None,
    reduction="mean",
):
    """
    Computes Precision, Recall, and AUC-PR for matched points across batches.

    Args:
        source_pixels_matched (torch.Tensor): Source points, shape (N, 2).
        target_pixels_matched (torch.Tensor): Predicted target points, shape (N, 2).
        true_pixels_matched (torch.Tensor): Ground truth target points, shape (N, 2) or None.
        batch_indexes (torch.Tensor): Batch indices for each point, shape (N,).
        confidence_scores (torch.Tensor): Confidence scores for each prediction, shape (N,).
        threshold (float): Distance threshold in pixels to consider a prediction as True Positive.
        fundamental (torch.Tensor, optional): Batch of fundamental matrices (shape: Bx3x3).
                                                      Required if true_pixels_matched is None.
        reduction (str): Reduction method ('mean' or 'none')

    Returns:
        If reduction='mean':
            precision (float): Mean Precision across all batches.
            recall (float): Mean Recall across all batches.
            auc_pr (float): Area Under the Precision-Recall Curve.
        If reduction='none':
            precision (torch.Tensor): Precision for each batch.
            recall (torch.Tensor): Recall for each batch.
            auc_pr (float): Overall Area Under the Precision-Recall Curve.
    """
    if true_pixels_matched is not None:
        distances = torch.norm(target_pixels_matched - true_pixels_matched, dim=1)
    else:
        if fundamental is None:
            raise ValueError(
                "Fundamental matrices must be provided when true_pixels_matched is None"
            )
        distances = epipolar_error(
            source_pixels_matched,
            target_pixels_matched,
            fundamental,
            batch_indexes,
            reduction="none",
        )

    labels = (distances < threshold).int()

    if reduction == "mean":
        sorted_scores, indices = torch.sort(confidence_scores, descending=True)
        sorted_labels = labels[indices]

        cumsum_tp = torch.cumsum(sorted_labels, dim=0)
        total_positives = labels.sum()
        total_predictions = len(labels)

        precision_curve = cumsum_tp / torch.arange(
            1, total_predictions + 1, device=source_pixels_matched.device
        )
        recall_curve = cumsum_tp / (total_positives + 1e-8)

        auc_pr = torch.trapz(precision_curve, recall_curve)

        mean_precision = precision_curve.mean()
        mean_recall = recall_curve.mean()

        return mean_precision.item(), mean_recall.item(), auc_pr.item()
    else:
        unique_batches = torch.unique(batch_indexes)
        precisions = []
        recalls = []

        for batch_id in unique_batches:
            batch_mask = batch_indexes == batch_id
            batch_labels = labels[batch_mask]
            batch_scores = confidence_scores[batch_mask]

            if batch_labels.sum() == 0:
                precisions.append(0.0)
                recalls.append(0.0)
                continue

            sorted_scores, indices = torch.sort(batch_scores, descending=True)
            sorted_labels = batch_labels[indices]

            cumsum_tp = torch.cumsum(sorted_labels, dim=0)
            total_positives = batch_labels.sum()
            total_predictions = len(batch_labels)

            precision_curve = cumsum_tp / torch.arange(
                1, total_predictions + 1, device=source_pixels_matched.device
            )
            recall_curve = cumsum_tp / (total_positives + 1e-8)

            precisions.append(precision_curve.mean().item())
            recalls.append(recall_curve.mean().item())

        precision_tensor = torch.tensor(precisions, device=source_pixels_matched.device)
        recall_tensor = torch.tensor(recalls, device=source_pixels_matched.device)

        sorted_scores, indices = torch.sort(confidence_scores, descending=True)
        sorted_labels = labels[indices]
        cumsum_tp = torch.cumsum(sorted_labels, dim=0)
        total_positives = labels.sum()
        total_predictions = len(labels)
        precision_curve = cumsum_tp / torch.arange(
            1, total_predictions + 1, device=source_pixels_matched.device
        )
        recall_curve = cumsum_tp / (total_positives + 1e-8)
        auc_pr = torch.trapz(precision_curve, recall_curve)

        return precision_tensor, recall_tensor, auc_pr.item()


def f1_score(points1, points2, F, batch_index, threshold=1.0, reduction="mean"):
    """
    Calculate F1-score for point matches based on epipolar constraint.

    A match is considered a True Positive if its distance to the
    corresponding ground truth epipolar line is less than the threshold (default: 1 pixel).

    Args:
        points1 (torch.Tensor): First set of points (shape: Nx2)
        points2 (torch.Tensor): Second set of points (shape: Nx2)
        F (torch.Tensor): Batch of fundamental matrices (shape: Bx3x3)
        batch_index (torch.Tensor): Batch indices for each point pair (shape: Nx1)
        threshold (float, optional): Threshold for considering a match as correct (default: 1.0)
        reduction (str): Reduction method ('mean' or 'none')

    Returns:
        If reduction='mean':
            float: Mean F1-score across all batches
        If reduction='none':
            torch.Tensor: Per-batch F1-scores (shape: B)
    """
    errors = epipolar_error(points1, points2, F, batch_index, reduction="none")

    correct_matches = errors < threshold

    batch_size = F.shape[0]

    if reduction == "none":
        batch_f1 = torch.zeros(batch_size, device=points1.device)

        for b in range(batch_size):
            batch_mask = batch_index.squeeze(-1) == b

            if not batch_mask.any():
                batch_f1[b] = 0.0
                continue

            batch_correct = correct_matches[batch_mask]
            batch_tp = batch_correct.sum().float()
            batch_total = batch_mask.sum().float()

            precision = (
                batch_tp / batch_total
                if batch_total > 0
                else torch.tensor(0.0, device=points1.device)
            )

            recall = precision

            if precision + recall > 0:
                batch_f1[b] = 2 * (precision * recall) / (precision + recall)

        return batch_f1

    else:  # reduction == "mean"
        true_positives = correct_matches.sum().item()
        total_predictions = len(errors)

        precision = true_positives / total_predictions if total_predictions > 0 else 0.0

        recall = precision

        f1 = (
            2 * (precision * recall) / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        return f1


def compute_metrics(
    mp,
    source_pixels_matched,
    target_pixels_matched,
    true_pixels_matched,
    batch_idx_match,
    scores,
    fundamental_pred,
    fundamental_gt,
):
    """
    Compute various metrics for evaluating matching performance.

    Args:
        mp: Matcher instance with config
        source_pixels_matched: Source pixel coordinates
        target_pixels_matched: Predicted target pixel coordinates
        true_pixels_matched: Ground truth target pixel coordinates
        batch_idx_match: Batch indices for each match
        scores: Confidence scores for each match
        fundamental_pred: Predicted fundamental matrix
        fundamental_gt: Ground truth fundamental matrix (may be zeros if GT poses unavailable)

    Returns:
        dict: Dictionary containing all computed metrics
    """
    fundamental_gt_norm = torch.norm(fundamental_gt.reshape(fundamental_gt.shape[0], -1), dim=1)
    has_valid_fundamental_gt = torch.all(fundamental_gt_norm > 1e-6).item()

    fundamental_for_epipolar = fundamental_gt if has_valid_fundamental_gt else fundamental_pred

    precision_val, recall_val, AUCPR = precision_recall(
        source_pixels_matched.detach(),
        target_pixels_matched.detach(),
        true_pixels_matched.detach() if true_pixels_matched is not None else None,
        batch_idx_match.detach(),
        scores.detach(),
        mp.config.MAX_EPIPOLAR_DISTANCE,
        fundamental_for_epipolar,
    )

    epipolar = epipolar_error(
        source_pixels_matched.cpu(),
        target_pixels_matched.cpu(),
        fundamental_pred.cpu(),
        batch_idx_match.cpu(),
    )

    if has_valid_fundamental_gt:
        fundamental = fundamental_error(fundamental_pred.cpu(), fundamental_gt.cpu())[0]
    else:
        fundamental = None

    if true_pixels_matched is not None:
        mean_match_distance = mean_matching_distance(
            target_pixels_matched.cpu(),
            true_pixels_matched.cpu(),
            batch_idx_match.cpu(),
        )
    else:
        mean_match_distance = None

    refine_metrics = refinement_metrics(
        mp,
        true_pixels_matched=true_pixels_matched,
    )

    return {
        "Precision": precision_val,
        "Recall": recall_val,
        "AUCPR": AUCPR,
        "EpipolarError": epipolar,
        "FundamentalError": fundamental,
        "MDistMean": mean_match_distance,
        **refine_metrics,
    }
