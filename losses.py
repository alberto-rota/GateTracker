# -------------------------------------------------------------------------------------------------#

""" Copyright (c) 2024 Asensus Surgical """

""" Code Developed by: Alberto Rota """
""" Supervision: Uriya Levy, Gal Weizman, Stefano Pomati """

# -------------------------------------------------------------------------------------------------#

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
import numpy as np
from rich.tree import Tree
from rich import print as rprint


# -------------------------------------------------------------------------------------------
# BASE LOSS FUNCTIONS
# These loss functions are the lowest leveels one and take in inputs directly. They are not
# combinations of other losses
# -------------------------------------------------------------------------------------------
class TripletLoss(nn.Module):
    """
    Triplet loss for learning discriminative embeddings.
    
    This loss function implements the standard triplet loss used in metric learning.
    It encourages embeddings of anchor-positive pairs to be closer together than
    anchor-negative pairs by a specified margin. The loss is computed using cosine
    similarity and Euclidean distance in the embedding space.
    
    The loss function is commonly used in self-supervised learning for learning
    meaningful feature representations without explicit labels.
    """
    
    def __init__(self, margin: float = 1.0) -> None:
        """
        Initialize the TripletLoss module.
        
        Args:
            margin: The margin by which anchor-positive distance should be smaller
                   than anchor-negative distance. Defaults to 1.0.
        """
        super(TripletLoss, self).__init__()
        self.margin = margin

    def dist(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Compute the distance between two sets of embeddings.
        
        Args:
            A: First set of embeddings
            B: Second set of embeddings
            
        Returns:
            Euclidean distance between the embeddings
        """
        cosine_sim = nn.functional.cosine_similarity(A, B)
        return torch.sqrt(torch.clamp(2 - 2 * cosine_sim, min=0))

    def forward(self, A: torch.Tensor, P: torch.Tensor, N: torch.Tensor) -> torch.Tensor:
        """
        Compute the triplet loss.
        
        Args:
            A: Anchor embeddings
            P: Positive embeddings (same class as anchors)
            N: Negative embeddings (different class from anchors)
            
        Returns:
            Mean triplet loss value
        """
        return torch.mean(
            torch.clamp(self.margin + self.dist(A, P) - self.dist(A, N), min=0)
        )


class InfoNCELoss(nn.Module):
    """
    Masked symmetric InfoNCE loss for patch correspondences.
    """

    def __init__(self, temperature: float = 0.07, symmetric: bool = True) -> None:
        super().__init__()
        self.temperature = temperature
        self.symmetric = symmetric
        self.eps = 1e-8

    def _masked_directional_loss(
        self,
        logits: torch.Tensor,
        positive_indices: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: [M, K] cosine similarities before temperature scaling
            positive_indices: [M] positive column index per row
            valid_mask: [M, K] bool mask for valid negatives/positives
            weights: [M] confidence weights in [0, 1]
        """
        if logits.numel() == 0:
            return logits.sum() * 0.0

        logits = logits / self.temperature
        if valid_mask is None:
            valid_mask = torch.ones_like(logits, dtype=torch.bool)
        else:
            valid_mask = valid_mask.bool()

        positive_mask = F.one_hot(
            positive_indices.long(), num_classes=logits.shape[-1]
        ).bool()
        valid_mask = valid_mask | positive_mask

        masked_logits = logits.masked_fill(
            ~valid_mask,
            torch.finfo(logits.dtype).min,
        )
        log_probs = masked_logits - torch.logsumexp(
            masked_logits, dim=-1, keepdim=True
        )
        losses = -log_probs.gather(
            dim=1, index=positive_indices.long().unsqueeze(1)
        ).squeeze(1)

        if weights is None:
            return losses.mean()

        weights = weights.float().clamp_min(0.0)
        normalizer = weights.sum().clamp_min(self.eps)
        return (losses * weights).sum() / normalizer

    def forward(
        self,
        source_to_target_logits: torch.Tensor,
        source_to_target_positive: torch.Tensor,
        source_to_target_mask: Optional[torch.Tensor] = None,
        target_to_source_logits: Optional[torch.Tensor] = None,
        target_to_source_positive: Optional[torch.Tensor] = None,
        target_to_source_mask: Optional[torch.Tensor] = None,
        contrastive_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        source_loss = self._masked_directional_loss(
            logits=source_to_target_logits,
            positive_indices=source_to_target_positive,
            valid_mask=source_to_target_mask,
            weights=contrastive_weights,
        )
        if (
            not self.symmetric
            or target_to_source_logits is None
            or target_to_source_positive is None
        ):
            return source_loss

        target_loss = self._masked_directional_loss(
            logits=target_to_source_logits,
            positive_indices=target_to_source_positive,
            valid_mask=target_to_source_mask,
            weights=contrastive_weights,
        )
        return 0.5 * (source_loss + target_loss)


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


class InliersLoss(nn.Module):
    """
    Loss function based on inlier classification for fundamental matrix estimation.
    
    This loss computes an F1 score based on inlier classification, where inliers
    are points that satisfy the epipolar constraint within a specified threshold.
    The loss encourages the model to predict fundamental matrices that maximize
    the number of inlier correspondences.
    """
    
    def __init__(self, threshold: float = 0.75) -> None:
        """
        Initialize the InliersLoss module.
        
        Args:
            threshold: Distance threshold for considering a point as an inlier.
                      Defaults to 0.75.
        """
        super(InliersLoss, self).__init__()
        self.threshold = threshold
        self.eps = 1e-12

    def forward(self, F_pred: torch.Tensor, F_gt: torch.Tensor, pts1: torch.Tensor, pts2: torch.Tensor) -> torch.Tensor:
        """
        Compute F1 score based on inlier classification.

        Args:
            pts1: Tensor of shape (B, N, 2) - pixel coordinates in image 1
            pts2: Tensor of shape (B, N, 2) - pixel coordinates in image 2
            F_pred: Tensor of shape (B, 3, 3) - predicted fundamental matrix
            F_gt: Tensor of shape (B, 3, 3) - ground truth fundamental matrix
            
        Returns:
            F1 score loss (1 - F1_score)
        """
        # Convert pixel coordinates to homogeneous coordinates
        batch_size, num_pts, _ = pts1.size()
        hom_pts1 = torch.cat(
            [pts1, torch.ones(batch_size, num_pts, 1, device=pts1.device)], dim=2
        )  # (B, N, 3)
        hom_pts2 = torch.cat(
            [pts2, torch.ones(batch_size, num_pts, 1, device=pts2.device)], dim=2
        )  # (B, N, 3)

        def epipolar_error(hom_pts1: torch.Tensor, hom_pts2: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
            """Compute symmetric epipolar error for a batch."""
            Ft_pts2 = torch.bmm(
                F.transpose(1, 2), hom_pts2.transpose(1, 2)
            )  # (B, 3, N)
            F_pts1 = torch.bmm(F, hom_pts1.transpose(1, 2))  # (B, 3, N)

            res = 1 / (Ft_pts2[:, :2, :].norm(dim=1) + self.eps)  # (B, N)
            res += 1 / (F_pts1[:, :2, :].norm(dim=1) + self.eps)  # (B, N)
            res *= torch.abs(
                torch.sum(
                    hom_pts2 * torch.bmm(F, hom_pts1.transpose(1, 2)).transpose(1, 2),
                    dim=2,
                )
            )  # (B, N)
            return res

        # Compute epipolar errors
        est_res = epipolar_error(hom_pts1, hom_pts2, F_pred)  # (B, N)
        gt_res = epipolar_error(hom_pts1, hom_pts2, F_gt)  # (B, N)

        # Determine inliers
        est_inliers = est_res < self.threshold  # (B, N)
        gt_inliers = gt_res < self.threshold  # (B, N)
        true_positives = est_inliers & gt_inliers  # (B, N)

        gt_inlier_count = gt_inliers.float().sum(dim=1)  # (B,)
        est_inlier_count = est_inliers.float().sum(dim=1)  # (B,)
        true_positive_count = true_positives.float().sum(dim=1)  # (B,)

        # Precision and recall
        precision = true_positive_count / (est_inlier_count + self.eps)  # (B,)
        recall = true_positive_count / (gt_inlier_count + self.eps)  # (B,)

        # F1 score
        f1_score = 2 * precision * recall / (precision + recall + self.eps)  # (B,)

        return 1 - f1_score.mean()


class SoftRankLoss(nn.Module):
    """
    Soft rank loss for fundamental matrix estimation.
    
    This loss function encourages the fundamental matrix to have proper rank-2
    structure by penalizing the log-determinant of F^T F. The loss helps ensure
    that the predicted fundamental matrix satisfies the mathematical constraints
    of a valid fundamental matrix.
    """

    def __init__(self) -> None:
        """Initialize the SoftRankLoss module."""
        super(SoftRankLoss, self).__init__()

    def forward(self, F: torch.Tensor, delta: float = 1e-6) -> torch.Tensor:
        """
        Compute the soft rank loss.
        
        Args:
            F: Fundamental matrix tensor of shape (B, 3, 3)
            delta: Small value added for numerical stability. Defaults to 1e-6.
            
        Returns:
            Log-determinant loss value
        """
        # Ensure F is positive definite by computing F^T F
        F_t_F = torch.bmm(F.transpose(1, 2), F)

        # Add delta * I to ensure numerical stability
        batch_size, n, _ = F_t_F.shape
        identity = delta * torch.eye(n, device=F.device).unsqueeze(0).expand(
            batch_size, -1, -1
        )
        F_t_F_stable = F_t_F + identity

        # Compute log-det loss
        log_det = torch.linalg.slogdet(F_t_F_stable).logabsdet
        loss = torch.sum(log_det)

        return loss


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) loss for image reconstruction.
    
    This loss function computes the SSIM between a target image and a warped/reconstructed
    image. SSIM measures the perceived quality of images by considering structural
    information, making it more suitable for image reconstruction tasks than
    simple pixel-wise losses like MSE.
    
    Note:
        Copyright of this class fully belongs to Shuwei Shao @ https://github.com/ShuweiShao/AF-SfMLearner
    """

    def __init__(self) -> None:
        """Initialize the SSIMLoss module."""
        super(SSIMLoss, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01**2
        self.C2 = 0.03**2

    def forward(self, target: torch.Tensor, warped: torch.Tensor) -> torch.Tensor:
        """
        Compute the SSIM loss between target and warped images.
        
        Args:
            target: Target image tensor
            warped: Warped/reconstructed image tensor
            
        Returns:
            SSIM loss value (1 - SSIM)
        """
        x = self.refl(target)
        y = self.refl(warped)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x**2) - mu_x**2
        sigma_y = self.sig_y_pool(y**2) - mu_y**2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x**2 + mu_y**2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1).mean()

    def __str__(self) -> str:
        """Return string representation of the loss."""
        return "SSIMLoss()"


class L1Loss(nn.Module):
    """
    L1 (Mean Absolute Error) loss for image reconstruction.
    
    This loss function computes the mean absolute difference between a target image
    and a warped/reconstructed image. L1 loss is robust to outliers and provides
    stable gradients for training.
    """

    def __init__(self) -> None:
        """Initialize the L1Loss module."""
        super(L1Loss, self).__init__()

    def forward(self, target: torch.Tensor, warped: torch.Tensor) -> torch.Tensor:
        """
        Compute the L1 loss.

        Args:
            target: Target image tensor
            warped: Warped/reconstructed image tensor

        Returns:
            L1 loss value
        """
        l1_loss = torch.mean(torch.abs(target - warped))
        return l1_loss

    def __str__(self) -> str:
        """Return string representation of the loss."""
        return "L1Loss()"


class EASLoss(nn.Module):
    """
    Edge-Aware Smoothness (EAS) loss for depth estimation.
    
    This loss function encourages smooth depth maps while preserving edges in the
    image. It applies higher smoothness penalties in regions with low image gradients
    and lower penalties near edges, helping to produce more realistic depth maps.
    """

    def __init__(self, alpha: int = 1) -> None:
        """
        Initialize the EAS loss module.
        
        Args:
            alpha: Weight factor for the smoothness loss. Defaults to 1.
        """
        super(EASLoss, self).__init__()
        self.alpha = alpha

    def forward(self, warped: torch.Tensor, depthmap: torch.Tensor) -> torch.Tensor:
        """
        Compute the EAS loss.

        Args:
            warped: Warped image tensor
            depthmap: Depth map tensor

        Returns:
            EAS loss value
        """
        # Calculate horizontal and vertical gradients of the disparity map
        # Calculate horizontal and vertical gradients of the image
        disparity_gradients_x = torch.abs(depthmap[:, :, :-1] - depthmap[:, :, 1:])
        disparity_gradients_y = torch.abs(depthmap[:, :-1, :] - depthmap[:, 1:, :])

        image_gradients_x = torch.mean(
            torch.abs(warped[:, :, :, :-1] - warped[:, :, :, 1:]), dim=1
        )  # Resulting shape: Bx1x448x447
        image_gradients_y = torch.mean(
            torch.abs(warped[:, :, :-1, :] - warped[:, :, 1:, :]), dim=1
        )  # Resulting shape: Bx1x447x448
        # Adjust disparity gradients to match the shape of image gradients
        disparity_gradients_x = disparity_gradients_x[:, :-1, :]  # Shape: Bx447x447
        disparity_gradients_y = disparity_gradients_y[:, :, :-1]  # Shape: Bx447x447

        # Match the dimensions for smoothness loss calculation
        image_gradients_x = image_gradients_x[:, :-1, :]  # Shape: Bx1x447x447
        image_gradients_y = image_gradients_y[:, :, :-1]  # Shape: Bx1x447x447

        # Calculate the smoothness loss
        smoothness_loss = disparity_gradients_x * torch.exp(
            -image_gradients_x
        ) + disparity_gradients_y * torch.exp(-image_gradients_y)

        # Calculate the average loss
        smoothness_loss = torch.mean(smoothness_loss)

        # Scale the loss by alpha
        smoothness_loss = self.alpha * smoothness_loss

        return smoothness_loss

    def __str__(self) -> str:
        """Return string representation of the loss."""
        return f"EASLoss(alpha={self.alpha})"


class ScaleLoss(nn.Module):
    """
    Scale loss for pose estimation.
    
    This loss function measures the difference between ground truth and predicted
    poses, typically used to ensure that the predicted camera motion has the
    correct scale relative to the ground truth.
    """

    def __init__(self) -> None:
        """Initialize the ScaleLoss module."""
        super(ScaleLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pose_gt: torch.Tensor, pose_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute the scale loss.

        Args:
            pose_gt: Ground truth pose tensor
            pose_pred: Predicted pose tensor

        Returns:
            Scale loss value
        """
        # Calculate the translation scale
        return self.mse(pose_gt, pose_pred)

    def __str__(self) -> str:
        """Return string representation of the loss."""
        return "ScaleLoss()"


# -------------------------------------------------------------------------------------------
# COMBINATION LOSS FUNCTIONS
# These loss functions are linear combinations of other losses
# -------------------------------------------------------------------------------------------
@dataclass
class LossComponent:
    """
    Dataclass to store information about a loss component and its required parameters.
    
    This class encapsulates all the information needed for a single loss component
    in a weighted combination loss, including the loss function, weight, required
    parameters, and optional decay rate.
    """

    name: str
    loss_fn: nn.Module
    weight: float
    required_params: Set[str]
    decay_rate: Optional[float] = None

    @property
    def current_weight(self) -> float:
        """Get the current weight after decay."""
        return self.weight


class WeightedCombinationLoss(nn.Module):
    """
    A loss module that combines multiple loss functions with weights.
    
    This class provides a flexible framework for combining multiple loss functions
    with configurable weights. It supports weight decay over time and provides
    detailed breakdown of individual loss components for monitoring and debugging.
    
    The loss module can handle nested loss functions and provides comprehensive
    logging and visualization capabilities.
    """

    def __init__(
        self,
        components: List[Tuple[str, nn.Module, float, Set[str]]],
        decay_config: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialize the WeightedCombinationLoss module.
        
        Args:
            components: List of tuples containing (name, loss_fn, weight, required_params)
                       for each loss component
            decay_config: Optional dictionary mapping component names to decay rates
        """
        super(WeightedCombinationLoss, self).__init__()

        # Normalize weights to sum to 1
        total_weight = sum(weight for _, _, weight, _ in components)
        decay_config = decay_config or {}

        # Create loss components
        self.components = [
            LossComponent(
                name=name,
                loss_fn=loss_fn,
                weight=weight,  # / total_weight,
                required_params=required_params,
                decay_rate=decay_config.get(name),
            )
            for name, loss_fn, weight, required_params in components
        ]

        # Store all required parameters
        self.all_required_params = set().union(
            *(comp.required_params for comp in self.components)
        )

        self.step_count = 0

    def forward(self, **kwargs: Any) -> torch.Tensor:
        """
        Compute the combined loss from all components.
        
        Args:
            **kwargs: Keyword arguments containing all required parameters for the loss components
            
        Returns:
            Combined weighted loss value
            
        Raises:
            ValueError: If any required parameters are missing
        """
        # Validate that all required parameters are provided
        missing_params = self.all_required_params - set(kwargs.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")

        total_loss = 0.0

        for component in self.components:
            # Extract only the arguments needed for this specific loss function
            fn_args = {k: kwargs[k] for k in component.required_params}

            loss = component.loss_fn(**fn_args)
            total_loss += loss * component.current_weight

        return total_loss

    def get_dict(self, prepend_tonames: str = "", **kwargs: Any) -> Dict[str, float]:
        """
        Get detailed breakdown of all loss components.
        
        Args:
            prepend_tonames: String to prepend to all loss names in the output dictionary
            **kwargs: Keyword arguments containing all required parameters
            
        Returns:
            Dictionary mapping loss component names to their individual loss values
            
        Raises:
            ValueError: If any required parameters are missing
        """
        # Validate parameters first
        missing_params = self.all_required_params - set(kwargs.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")

        results = {}
        total_loss = 0.0

        for component in self.components:
            component_loss = 0.0
            currentcomponentname = f"{prepend_tonames}{component.name}"
            fn_args = {k: kwargs[k] for k in component.required_params}

            # Handle nested loss functions that have their own get_dict
            if hasattr(component.loss_fn, "get_dict"):
                sub_losses = component.loss_fn.get_dict(
                    prepend_tonames=f"{prepend_tonames}{component.name}/", **fn_args
                )
                results.update(sub_losses)
                for c in component.loss_fn.components:
                    fn_args = {k: kwargs[k] for k in c.required_params}
                    loss = c.loss_fn(**fn_args)
                    component_loss += loss * c.current_weight
                results[f"{prepend_tonames}{component.name}"] = component_loss.item()

            else:
                loss = component.loss_fn(**fn_args)
                loss_value = loss.item()
                results[f"{prepend_tonames}{component.name}"] = loss_value

        return results

    def get_weights(self, prepend_tonames: str = "") -> Dict[str, float]:
        """
        Recursively get a dictionary of weights for all components.

        Args:
            prepend_tonames: String to prepend to all loss names in the output dictionary

        Returns:
            Dictionary mapping loss component names to their weights
        """
        weights = {}
        for component in self.components:
            component_name = f"{prepend_tonames}{component.name}"
            if isinstance(component.loss_fn, WeightedCombinationLoss):
                # Recurse into nested WeightedCombinationLoss
                sub_weights = component.loss_fn.get_weights(
                    prepend_tonames=f"{component_name}_"
                )
                weights.update(sub_weights)
            weights[component_name] = component.current_weight
        return weights

    def step(self) -> None:
        """
        Update step count and decay weights if configured, recursively stepping child components.
        
        This method should be called after each training step to update the step count
        and apply any configured weight decay to the loss components.
        """
        # print("Stepping", self.__class__.__name__)
        self.step_count += 1
        for component in self.components:
            # Decay the weight if a decay rate is specified
            if component.decay_rate:
                component.weight *= np.exp(-component.decay_rate * self.step_count)
            # Recursively call step on nested WeightedCombinationLoss
            if hasattr(component.loss_fn, "step"):
                component.loss_fn.step()

    def __str__(self) -> str:
        """Create a hierarchical string representation of the loss structure."""
        components_str = []
        for comp in self.components:
            weight_str = f"{comp.current_weight:.3f}"
            if comp.decay_rate:
                weight_str += f" (decaying @ {comp.decay_rate:.2e})"

            loss_str = str(comp.loss_fn).replace("\n", "\n    ")

            params_str = f"params={sorted(comp.required_params)}"
            components_str.append(
                f"    {weight_str} * {loss_str} [{comp.name}]\n    └─ {params_str}"
            )

        return f"{self.__class__.__name__}(\n" + "\n".join(components_str) + "\n)"

    def rich_print(self, parent_tree: Optional[Tree] = None) -> None:
        """
        Print a rich tree visualization of the loss structure.
        
        Args:
            parent_tree: Optional parent tree node for nested visualization
        """
        if parent_tree is None:
            tree = Tree(f"{self.__class__.__name__}")
        else:
            tree = parent_tree.add(f"{self.__class__.__name__}")

        for comp in self.components:
            weight_str = f"{comp.current_weight:.3f}"
            if comp.decay_rate:
                weight_str += f" (decay: {comp.decay_rate:.2e})"

            branch = tree.add(
                f"[blue]{comp.name}[/blue]([cyan]{','.join(comp.required_params)}[/cyan]) × [yellow]{weight_str}[/yellow]"
            )

            # Add parameters branch

            # Add loss function branch
            if hasattr(comp.loss_fn, "rich_print"):
                comp.loss_fn.rich_print(branch)
            else:
                loss_str = str(comp.loss_fn).replace("\n", "\n    ")
                # branch.add(f"[green]{loss_str}[/green]")

        if parent_tree is None:
            rprint(tree)


class TrackingPositionLoss(nn.Module):
    r"""
    Visibility-masked Huber loss on predicted vs GT point positions.

    .. math::
        \mathcal{L}_{\text{pos}} = \frac{1}{\sum v_i^t}
        \sum_{i,t: v_i^t = 1} \text{Huber}(\hat{p}_i^t - p_i^t)

    Operates on tensors ``[B, Q, T, 2]`` where visibility ``[B, Q, T]`` serves
    as the weight mask.
    """

    def __init__(self, beta: float = 1.0, eps: float = 1e-8) -> None:
        super().__init__()
        self.beta = beta
        self.eps = eps

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        visibility: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            prediction: [B, Q, T, 2] predicted positions.
            target:     [B, Q, T, 2] GT positions.
            visibility: [B, Q, T]    GT visibility mask (bool or float).

        Returns:
            Scalar loss.
        """
        if prediction.numel() == 0:
            return prediction.sum() * 0.0

        vis = visibility.float()  # [B, Q, T]
        per_point = F.smooth_l1_loss(
            prediction, target, reduction="none", beta=self.beta,
        ).sum(dim=-1)  # [B, Q, T]

        normalizer = vis.sum().clamp_min(self.eps)
        return (per_point * vis).sum() / normalizer


class VisibilityLoss(nn.Module):
    r"""
    Binary cross-entropy on predicted visibility logits vs GT visibility.

    .. math::
        \mathcal{L}_{\text{vis}} = \text{BCE}(\sigma(\hat{v}_i^t),\; v_i^t)
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: [B, Q, T] raw visibility logits.
            target: [B, Q, T] GT visibility (bool or float 0/1).

        Returns:
            Scalar BCE loss.
        """
        if logits.numel() == 0:
            return logits.sum() * 0.0
        return F.binary_cross_entropy_with_logits(
            logits.float(), target.float(), reduction="mean",
        )


class MONO3D_Loss(WeightedCombinationLoss):
    """
    A flexible loss module that combines multiple loss functions with weights.
    
    This is the main loss class used in the MONO3D framework. It extends the
    WeightedCombinationLoss with additional functionality specific to the MONO3D
    training pipeline, including automatic total loss computation and enhanced
    logging capabilities.
    
    Supports weight decay and detailed loss reporting for comprehensive training
    monitoring and debugging.
    """

    def __init__(
        self,
        components: List[Tuple[str, nn.Module, float, Set[str]]],
        decay_config: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialize the MONO3D_Loss module.
        
        Args:
            components: List of tuples containing (name, loss_fn, weight, required_params)
                       for each loss component
            decay_config: Optional dictionary mapping component names to decay rates
        """
        super(MONO3D_Loss, self).__init__(components, decay_config)

    def get_dict(self, prepend_tonames: str = "", **kwargs: Any) -> Dict[str, float]:
        """
        Get detailed breakdown of all loss components including total loss.
        
        Args:
            prepend_tonames: String to prepend to all loss names in the output dictionary
            **kwargs: Keyword arguments containing all required parameters
            
        Returns:
            Dictionary mapping loss component names to their individual loss values,
            plus the total combined loss
        """
        results = super().get_dict(prepend_tonames, **kwargs)
        results[f"{prepend_tonames}Loss"] = super().forward(**kwargs).item()
        return results

    def __str__(self) -> str:
        """Return string representation of the loss structure."""
        return super().__str__()
