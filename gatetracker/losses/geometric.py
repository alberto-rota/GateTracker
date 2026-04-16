# -------------------------------------------------------------------------------------------------#

""" Copyright (c) 2024 Asensus Surgical """

""" Code Developed by: Alberto Rota """
""" Supervision: Uriya Levy, Gal Weizman, Stefano Pomati """

# -------------------------------------------------------------------------------------------------#

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class WeightedSmoothL1Loss(nn.Module):
    """
    Confidence-weighted Smooth L1 regression for fine correspondence refinement.
    """

    def __init__(self, beta: float = 1.0, eps: float = 1e-8) -> None:
        super().__init__()
        self.beta = beta
        self.eps = eps

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            prediction: [M, 2] predicted pixel coordinates.
            target: [M, 2] pseudo-GT pixel coordinates.
            weights: [M] optional confidence weights.
            valid_mask: [M] optional validity mask.

        Returns:
            Scalar weighted Smooth L1 loss.
        """
        if prediction.numel() == 0:
            return prediction.sum() * 0.0

        per_match_loss = F.smooth_l1_loss(
            prediction,
            target,
            reduction="none",
            beta=self.beta,
        ).sum(dim=1)  # [M]

        if valid_mask is None:
            valid_mask = torch.ones_like(per_match_loss, dtype=torch.bool)
        if not valid_mask.any():
            return per_match_loss.sum() * 0.0

        if weights is None:
            weights = valid_mask.float()
        else:
            weights = weights.float() * valid_mask.float()

        normalizer = weights.sum().clamp_min(self.eps)
        return (per_match_loss * weights).sum() / normalizer


class EpipolarLoss(nn.Module):
    """
    PyTorch module implementing robust symmetric epipolar distance loss.

    This loss computes the robust symmetric epipolar distance between corresponding points
    in two images given a predicted fundamental matrix. The loss is used to train
    models to predict accurate fundamental matrices by minimizing the geometric
    error between matched point correspondences.

    The symmetric epipolar distance measures how well the predicted fundamental matrix
    satisfies the epipolar constraint for the given point correspondences.
    """

    def __init__(self, gamma: float = 0.5) -> None:
        """
        Initialize the loss module.

        Args:
            gamma: Robust parameter for clamping the loss. Defaults to 0.5.
        """
        super().__init__()
        self.gamma = gamma

    def _symmetric_epipolar_distance(self, pts1: torch.Tensor, pts2: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
        """
        Compute symmetric epipolar distance.

        Args:
            pts1: Points in first image (B, N, 3)
            pts2: Points in second image (B, N, 3)
            F: Fundamental matrix (B, 3, 3)

        Returns:
            Symmetric epipolar distance (B, N)
        """

        # Normalize points to [-1, 1]
        pts1 = pts1 / 192 - 1
        pts2 = pts2 / 192 - 1
        # Convert points to homogeneous coordinates
        pts1 = torch.cat([pts1, torch.ones_like(pts1[:, :, :1])], dim=2)  # (B, N, 3)
        pts2 = torch.cat([pts2, torch.ones_like(pts2[:, :, :1])], dim=2)  # (B, N, 3)

        # Compute epipolar lines
        line1 = torch.bmm(pts1, F)  # (B, N, 3)
        line2 = torch.bmm(pts2, F.permute(0, 2, 1))  # (B, N, 3)

        # Compute scalar product
        scalar_product = (pts2 * line1).sum(2)  # (B, N)

        # Compute normalized distance
        norm_term = 1 / line1[:, :, :2].norm(2, 2) + 1 / line2[:, :, :2].norm(  # (B, N)
            2, 2
        )  # (B, N)

        return scalar_product.abs() * norm_term

    def forward(self, F_pred: torch.Tensor, pts1: torch.Tensor, pts2: torch.Tensor, gamma: Optional[float] = None) -> torch.Tensor:
        """
        Forward pass computing the loss.

        Args:
            F_pred: Predicted fundamental matrix (B, 3, 3)
            pts1: Points in first image (B, N, 3)
            pts2: Points in second image (B, N, 3)
            gamma: Override default gamma value. Defaults to None.

        Returns:
            Mean robust symmetric epipolar distance
        """
        if gamma is None:
            gamma = self.gamma

        # Compute symmetric epipolar distance
        sed = self._symmetric_epipolar_distance(pts1, pts2, F_pred)

        # Apply robust clamping
        loss = torch.clamp(sed, max=gamma)

        # Return mean loss
        return loss.mean()
