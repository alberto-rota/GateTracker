import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import tempfile

from utilities import *
from utilities.dev_utils import download_from_gcs
import networks.backbones as backbones
import networks.depth_decoding as depth_decoding

from matching.backbone import FeatureExtractor
from matching.helpers import (
    apply_refinement_offsets,
    filter_scores,
    points_to_patches,
)
from matching import learning, epipolar, refinement, correspondence, metrics, projections

import warnings
from logger import get_logger
from utilities.tensor_utils import (
    chw2embedding,
    embedding2chw,
    embedding_confidence_from_pixels,
    embedding_mask_from_pixels,
)
logger = get_logger(__name__).set_context("MATCHING")
torch.autograd.set_detect_anomaly(True)


class Matcher:
    def __init__(
        self,
        config,
        model=None,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        ### INITIALIZATIONS
        self.config = config
        self.device = device
        self.refinement_method = str(config.get("REFINEMENT_METHOD", "fft")).lower()
        self.latest_refinement_state = {}

        ### MODULES
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            if isinstance(model, MatcherModel):  # If model passed directly, use it
                self.model = model.to(self.device)
            elif (
                isinstance(model, str) and model != ""
            ):  # If model passed as a name str, load checkpoint
                self.model = MatcherModel(
                    shared_key="asdjnasljkn",
                    backbone_brand=config["BACKBONE_BRAND"],
                    size=config["BACKBONE_SIZE"],
                    resampled_patch_size=config["RESAMPLED_PATCH_SIZE"],
                    backbone_family=config.get("BACKBONE_FAMILY", "dinov2"),
                    dino_model_name=config.get("DINO_MODEL_NAME", None),
                    dino_layers=config.get("DINO_LAYERS", "auto"),
                    fusion_head=config.get(
                        "FUSION_HEAD", "register_gated_hierarchical"
                    ),
                    fusion_gate_temperature=config.get(
                        "FUSION_GATE_TEMPERATURE", 2.0
                    ),
                    fusion_uniform_mixing=config.get(
                        "FUSION_UNIFORM_MIXING", 0.1
                    ),
                    fusion_layer_dropout=config.get(
                        "FUSION_LAYER_DROPOUT", 0.1
                    ),
                    fusion_gate_logit_bound=config.get(
                        "FUSION_GATE_LOGIT_BOUND", 1.5
                    ),
                    refinement_method=config.get("REFINEMENT_METHOD", "fft"),
                    fine_feature_dim=config.get("FINE_REFINEMENT_DIM", 64),
                    fine_feature_stride=config.get("FINE_FEATURE_STRIDE", 4),
                    refinement_context_layer=config.get(
                        "FINE_REFINEMENT_CONTEXT_LAYER", "auto"
                    ),
                ).to(self.device)
                self.model.fromArtifact(model)
            elif (
                config.get("RUN") is None
                or config.get("RUN") == ""
                or model == ""
                or model is None
            ):  # If no model or model name provided, initalize from scratch.
                self.model = MatcherModel(
                    backbone_brand=config["BACKBONE_BRAND"],
                    size=config["BACKBONE_SIZE"],
                    resampled_patch_size=config["RESAMPLED_PATCH_SIZE"],
                    backbone_family=config.get("BACKBONE_FAMILY", "dinov2"),
                    dino_model_name=config.get("DINO_MODEL_NAME", None),
                    dino_layers=config.get("DINO_LAYERS", "auto"),
                    fusion_head=config.get(
                        "FUSION_HEAD", "register_gated_hierarchical"
                    ),
                    fusion_gate_temperature=config.get(
                        "FUSION_GATE_TEMPERATURE", 2.0
                    ),
                    fusion_uniform_mixing=config.get(
                        "FUSION_UNIFORM_MIXING", 0.1
                    ),
                    fusion_layer_dropout=config.get(
                        "FUSION_LAYER_DROPOUT", 0.1
                    ),
                    fusion_gate_logit_bound=config.get(
                        "FUSION_GATE_LOGIT_BOUND", 1.5
                    ),
                    refinement_method=config.get("REFINEMENT_METHOD", "fft"),
                    fine_feature_dim=config.get("FINE_REFINEMENT_DIM", 64),
                    fine_feature_stride=config.get("FINE_FEATURE_STRIDE", 4),
                    refinement_context_layer=config.get(
                        "FINE_REFINEMENT_CONTEXT_LAYER", "auto"
                    ),
                ).to(self.device)

        if config.get("TRACKING_HEAD", False):
            self.model.enable_tracking_head(
                window_radius=int(config.get("FINE_REFINEMENT_WINDOW_RADIUS", 2)),
                temperature=float(config.get("FINE_REFINEMENT_TEMPERATURE", 0.1)),
            )

        self.warp = projections.Warp(
            config.IMAGE_HEIGHT,
            config.IMAGE_WIDTH,
        ).to(self.device)
        self.RANSAC_unwrapped = epipolar.FundamentalEstimatorRANSAC()
        self.EIGHPA_raw = epipolar.FundamentalEstimator8PA()

        ### DIMENSIONS
        self.height, self.width, self.batch_size = (
            config.IMAGE_HEIGHT,
            config.IMAGE_WIDTH,
            config.BATCH_SIZE,
        )
        _, self.embed_dim, self.seq_len = self.model.embed(
            torch.randn(
                self.batch_size,
                3,
                self.height,
                self.width,
            ).to(self.device),
            mode="seq",
        ).shape
        self.patch_size = int((self.height * self.width / self.seq_len) ** 0.5)

    def synthethize_ground_truth(
        self,
        framestack,
        K,
        camera_pose_gt,
        depthstack=None,
        source_matched_points=None,
        batch_idx_match=None,
    ):
        if depthstack is None:
            depthstack = (
                1 / self.model.depth(framestack) * self.config.DEPTH_SCALE_FACTOR
            )
        if source_matched_points is None:
            source_matched_points = learning.generate_grid(
                num_points=self.config.TRIPLETS_TO_MINE,
                batch_size=framestack.shape[0],
                framestack=framestack,  # or image_height, image_width
                device=self.device,
            )

        warping_output = self.warp(
            framestack[:, 0],  # Source image
            depthstack,  # Depth map
            K,  # Camera intrinsics
            camera_pose_gt,  # Camera pose
            points_to_match=source_matched_points,
            batch_idx_match=batch_idx_match,
            return_mask=True,
            return_artifacts=True,
        )
        warped, target_matched_points_true, mask = (
            warping_output["warped"],
            warping_output["matches"],
            warping_output["mask"],
        )
        raw_warped = warping_output.get("raw", warped)
        raw_mask = warping_output.get("raw_mask")
        visibility_confidence = warping_output.get("visibility_confidence")
        if visibility_confidence is None:
            visibility_confidence = mask.float()

        photometric_beta = float(
            self.config.get("CONFIDENCE_PHOTOMETRIC_BETA", 6.0)
        )
        patch_valid_ratio = float(
            self.config.get("CONFIDENCE_PATCH_VALID_RATIO", 0.35)
        )
        target_image = framestack[:, -1]  # [B, 3, H, W]
        photometric_error = (raw_warped - target_image).abs().mean(
            dim=1, keepdim=True
        )  # [B, 1, H, W]
        photometric_confidence = torch.exp(
            -photometric_beta * photometric_error
        )  # [B, 1, H, W]
        pixel_confidence = visibility_confidence[:, :1] * photometric_confidence  # [B, 1, H, W]
        if raw_mask is not None:
            pixel_confidence = pixel_confidence * raw_mask[:, :1]
        pixel_confidence = pixel_confidence.clamp_(0.0, 1.0)  # [B, 1, H, W]
        pixel_confidence_rgb = pixel_confidence.expand(-1, 3, -1, -1)  # [B, 3, H, W]

        # Build patch-space validity and confidence from continuous warp confidence
        embedding_mask = chw2embedding(
            embedding_mask_from_pixels(
                pixel_confidence_rgb,
                patch_size=self.config.RESAMPLED_PATCH_SIZE,
                embedding_dim=self.embed_dim,
                min_valid_ratio=patch_valid_ratio,
            )
        )
        embedding_confidence = embedding_confidence_from_pixels(
            pixel_confidence_rgb,
            patch_size=self.config.RESAMPLED_PATCH_SIZE,
        )
        return {
            "warped": warped,
            "source_matched_points": source_matched_points,
            "target_matched_points_true": target_matched_points_true,
            "embedding_mask": embedding_mask,
            "embedding_confidence": embedding_confidence,
            "pixel_confidence": pixel_confidence,
            # "cloud": warping_output["cloud"],
            # "rgb_vec": warping_output["rgb_vec"],
        }

    def mine_triplets(
        self,
        modeloutput,
        source_matched_points,
        target_matched_points_true,
        embedding_mask,
        embedding_confidence=None,
    ):
        source_embs = modeloutput["source_embedding_match"]  # [B, C, HW?]
        target_embs = modeloutput["target_embedding_match"]
        source_matched_embs, source_matched_embs_idx, _ = points_to_patches(
            source_matched_points,
            embedding2chw(source_embs, embed_dim_last=False),
            patch_size=self.patch_size,
        )
        target_matched_embs, target_matched_embs_idx, embedding_match_mask = (
            points_to_patches(
                target_matched_points_true,
                embedding2chw(target_embs, embed_dim_last=False),
                patch_size=self.patch_size,
                mask=embedding2chw(embedding_mask, embed_dim_last=False),
            )
        )
        match_confidence = None
        if embedding_confidence is not None:
            match_confidence, _, _ = points_to_patches(
                target_matched_points_true,
                embedding_confidence,
                patch_size=self.patch_size,
            )
            match_confidence = match_confidence[:, 0]
        triplets_dict = learning.mine_triplets_optimized(
            sourceembs=source_matched_embs,
            targetembs=target_matched_embs,
            sourceembs_idx=source_matched_embs_idx,
            targetembs_idx=target_matched_embs_idx,
            target_mask=embedding_match_mask,
            match_confidence=match_confidence,
        )
        return triplets_dict

    def compute_correspondences(
        self,
        modeloutput,
        framestack,
        embedding_mask=None,
        knn=1,
        filter_inliers=False,
    ):
        """
        Compute correspondences between source and target patches using the model output and kNN.
        Args:
            modeloutput: Dictionary containing model outputs
            framestack: Stack of frames to process
            embedding_mask: Optional mask for embeddings (e.g., for occlusion)
            knn: Number of nearest neighbors to consider (default=1)
        """
        refinement_method = str(
            self.config.get("REFINEMENT_METHOD", self.refinement_method)
        ).lower()
        use_feature_refiner = refinement_method == "feature_softargmax"
        confidence_threshold = self.config.get("PIXEL_MATCHING_SCORE_THRESHOLD", None)

        # Extract embeddings and get initial matches
        (
            batch_idx_match,
            source_pixels_matched,
            target_pixels_matched,
            source_patches_matched,
            target_patches_matched,
            descriptor_scores,  # Renamed to descriptor_scores to clarify
            sim_matrix,
        ) = correspondence.get_matching_points_with_patches(
            modeloutput["source_embedding_match"],
            modeloutput["target_embedding_match"],
            framestack[:, 0],
            framestack[:, 1],
            threshold=self.config.PATCH_MATCHING_SCORE_THRESHOLD,
            min_matches=self.config.MIN_MATCHES_TO_COLLECT,
            max_matches=self.config.MAX_MATCHES_TO_PROCESS,
            embedding_mask=embedding_mask,
            patch_size=self.config.RESAMPLED_PATCH_SIZE,
            patch_size_enlarged=self.config.REFINEMENT_AREA,
            knn=knn,
            extract_patches=not use_feature_refiner,
        )
        if use_feature_refiner:
            refinement_feature_maps = getattr(
                self.model, "latest_refinement_feature_maps", {}
            )
            source_feature_map = refinement_feature_maps.get("source")
            target_feature_map = refinement_feature_maps.get("target")
            if source_feature_map is None or target_feature_map is None:
                raise RuntimeError(
                    "Feature refinement requested, but fine feature maps are unavailable."
                )
            refinement_window_radius = int(
                self.config.get("FINE_REFINEMENT_WINDOW_RADIUS", 2)
            )
            refinement_stride = int(
                self.config.get(
                    "FINE_FEATURE_STRIDE",
                    getattr(self.model, "fine_feature_stride", 4),
                )
            )
            source_pixel_offset, target_pixel_offset, refinement_scores = (
                refinement.feature_softargmax_refiner(
                    source_feature_map=source_feature_map,
                    target_feature_map=target_feature_map,
                    source_pixels=source_pixels_matched,
                    target_pixels=target_pixels_matched,
                    batch_indices=batch_idx_match,
                    window_radius=refinement_window_radius,
                    feature_stride=refinement_stride,
                    softmax_temperature=float(
                        self.config.get("FINE_REFINEMENT_TEMPERATURE", 0.1)
                    ),
                    confidence_threshold=confidence_threshold,
                )
            )
            refinement_patch_size = refinement.feature_refinement_patch_size(
                refinement_window_radius,
                refinement_stride,
            )
        else:
            source_pixel_offset, target_pixel_offset, refinement_scores = (
                refinement.FFT_patch_refiner(
                    source_patches_matched,
                    target_patches_matched,
                    patch_size=self.config.REFINEMENT_AREA,
                    confidence_threshold=confidence_threshold,
                )
            )
            refinement_patch_size = int(self.config.REFINEMENT_AREA)

        points_data = {
            "source_pixels": source_pixels_matched,
            "target_pixels": target_pixels_matched,
            "batch_idx": batch_idx_match,
            "source_patches": source_patches_matched,
            "target_patches": target_patches_matched,
            "source_offset": source_pixel_offset,
            "target_offset": target_pixel_offset,
        }

        # # Filter based on scores
        if filter_inliers:
            filtered_points, scores = filter_scores(points_data, descriptor_scores)

            # Extract filtered data
            source_pixels_matched = filtered_points["source_pixels"]
            target_pixels_matched = filtered_points["target_pixels"]
            batch_idx_match = filtered_points["batch_idx"]
            source_patches_matched = filtered_points["source_patches"]
            target_patches_matched = filtered_points["target_patches"]
            source_pixel_offset = filtered_points["source_offset"]
            target_pixel_offset = filtered_points["target_offset"]

        coarse_source_pixels = source_pixels_matched.clone()  # [M, 2]
        coarse_target_pixels = target_pixels_matched.clone()  # [M, 2]

        # Apply refinement offsets to the matched pixel coordinates
        source_pixels_matched, target_pixels_matched = apply_refinement_offsets(
            source_pixels_matched,
            target_pixels_matched,
            source_pixel_offset,
            target_pixel_offset,
            refinement_patch_size,
        )
        active_mask = (
            refinement_scores > 0
            if confidence_threshold is not None
            else torch.ones_like(refinement_scores, dtype=torch.bool)
        )  # [M]
        self.latest_refinement_state = {
            "method": refinement_method,
            "patch_size": refinement_patch_size,
            "source_offsets": source_pixel_offset,
            "target_offsets": target_pixel_offset,
            "scores": refinement_scores,
            "active_mask": active_mask,
            "coarse_source_pixels": coarse_source_pixels,
            "coarse_target_pixels": coarse_target_pixels,
            "refined_source_pixels": source_pixels_matched,
            "refined_target_pixels": target_pixels_matched,
            "batch_idx_match": batch_idx_match,
        }
        # Return the correspondence results
        return {
            "source_pixels_matched": source_pixels_matched,
            "target_pixels_matched": target_pixels_matched,
            "batch_idx_match": batch_idx_match,
            "descriptor_scores": descriptor_scores,  # Use our unified scores
            "refinement_scores": refinement_scores,  # Use our unified scores
            "sim_matrix": sim_matrix,
        }

    def _sample_point_values(
        self,
        value_map: torch.Tensor,
        points: torch.Tensor,
        batch_idx_match: torch.Tensor,
    ) -> torch.Tensor:
        """
        Samples scalar supervision values at arbitrary pixel coordinates.

        Args:
            value_map: [B, 1, H, W] scalar map.
            points: [M, 2] pixel coordinates in `(x, y)`.
            batch_idx_match: [M] batch index per point.

        Returns:
            [M] sampled values in `[0, 1]`.
        """
        if value_map is None or points.numel() == 0:
            return None

        selected_maps = value_map[batch_idx_match]  # [M, 1, H, W]
        grid = points.float().view(-1, 1, 1, 2).clone()  # [M, 1, 1, 2]
        grid[..., 0] = (grid[..., 0] / max(selected_maps.shape[-1] - 1, 1)) * 2 - 1
        grid[..., 1] = (grid[..., 1] / max(selected_maps.shape[-2] - 1, 1)) * 2 - 1
        sampled = F.grid_sample(
            selected_maps,
            grid,
            mode="bilinear",
            align_corners=True,
        )
        return sampled.squeeze(1).squeeze(-1).squeeze(-1).clamp(0.0, 1.0)  # [M]

    def compute_refinement_loss(
        self,
        target_pixels_matched: torch.Tensor,
        target_pixels_true: torch.Tensor,
        batch_idx_match: torch.Tensor,
        pixel_confidence: torch.Tensor = None,
        loss_fn=None,
    ):
        """
        Computes weighted refinement supervision on pixel correspondences.

        Args:
            target_pixels_matched: [M, 2] refined target pixels.
            target_pixels_true: [M, 2] pseudo-GT target pixels.
            batch_idx_match: [M] batch index per correspondence.
            pixel_confidence: [B, 1, H, W] pseudo-GT confidence map.
            loss_fn: Weighted regression loss module.

        Returns:
            Dict with the scalar loss and supervision statistics.
        """
        zero = (
            target_pixels_matched.sum() * 0.0
            if isinstance(target_pixels_matched, torch.Tensor)
            else torch.tensor(0.0, device=self.device)
        )
        if (
            loss_fn is None
            or target_pixels_matched is None
            or target_pixels_true is None
            or target_pixels_matched.numel() == 0
            or str(self.config.get("REFINEMENT_METHOD", self.refinement_method)).lower()
            != "feature_softargmax"
        ):
            return {"loss": zero, "active_matches": 0, "weight_mean": 0.0}

        supervision_weights = self._sample_point_values(
            pixel_confidence,
            target_pixels_true.detach(),
            batch_idx_match,
        )
        valid_mask = torch.isfinite(target_pixels_matched).all(dim=1) & torch.isfinite(
            target_pixels_true
        ).all(dim=1)
        if supervision_weights is not None:
            valid_mask = valid_mask & (supervision_weights > 0)

        if not valid_mask.any():
            return {"loss": zero, "active_matches": 0, "weight_mean": 0.0}

        loss = loss_fn(
            prediction=target_pixels_matched.float(),
            target=target_pixels_true.float(),
            weights=supervision_weights,
            valid_mask=valid_mask,
        )
        weight_mean = (
            float(supervision_weights[valid_mask].mean().item())
            if supervision_weights is not None
            else 1.0
        )
        return {
            "loss": loss,
            "active_matches": int(valid_mask.sum().item()),
            "weight_mean": weight_mean,
        }

    def RANSAC(
        self,
        source_pixels_matched,
        target_pixels_matched,
        batch_idx_match,
    ):
        """
        Run RANSAC to estimate fundamental matrix and identify inliers.

        Args:
            source_pixels_matched: Source pixel coordinates of shape [N, 2]
            target_pixels_matched: Target pixel coordinates of shape [N, 2]
            batch_idx_match: Batch indices of shape [N]

        Returns:
            F: Estimated fundamental matrix of shape [B, 3, 3]
            inliers: Boolean tensor of shape [N] indicating inliers
            updated_scores: If scores provided, returns scores updated with epipolar information
        """
        # Run RANSAC to estimate fundamental matrix
        F, inliers, _ = self.RANSAC_unwrapped(
            source_pixels_matched,
            target_pixels_matched,
            batch_idx_match,
            max_epipolar_distance=self.config.MAX_EPIPOLAR_DISTANCE,
        )

        epipolar_errors = metrics.epipolar_error(
            source_pixels_matched,
            target_pixels_matched,
            F,
            batch_idx_match,
            reduction="none",
        )
        # Convert errors to scores (1.0 for error=0, decreasing as error increases)
        epipolar_scores = torch.exp(
            -epipolar_errors / self.config.MAX_EPIPOLAR_DISTANCE
        )

        return {
            "F": F,
            "inliers": inliers,
            "scores": epipolar_scores,
        }

    def EightPointAlgorithm(
        self, source_pixels_matched, target_pixels_matched, batch_idx_match, scores
    ):
        F = self.eightPA(
            source_pixels_matched.float(),
            target_pixels_matched.float(),
            scores.float(),
            batch_idx_match,
        )
        return {"F": F}

    # def compute_metrics(
    #     self,
    #     source_pixels_matched,
    #     target_pixels_matched,
    #     true_pixels_matched,
    #     batch_idx_match,
    #     scores,
    #     fundamental_pred,
    #     fundamental_gt,
    # ):
    #     """
    #     Compute various metrics for evaluating matching performance.

    #     Args:
    #         target_pixels_matched: Predicted target pixel coordinates
    #         true_pixels_matched: Ground truth target pixel coordinates
    #         batch_idx_match: Batch indices for each match
    #         scores: Confidence scores for each match
    #         fundamental_pred: Predicted fundamental matrix
    #         fundamental_pred8: Predicted fundamental matrix from 8-point algorithm
    #         fundamental_gt: Ground truth fundamental matrix (may be zeros if GT poses unavailable)
    #         inliers: Inlier mask
    #         triplets_dict: Dictionary containing triplet information
    #         loss_tensor: Loss tensor
    #         lossdfe: DFE loss tensor
    #         patch_size: Patch size used for matching
    #         inlier_patch_ratio: Ratio for determining inliers
    #         gradient_accumulation_steps: Number of gradient accumulation steps
    #         batch_size: Batch size

    #     Returns:
    #         dict: Dictionary containing all computed metrics
    #     """
    #     # Check if fundamental_gt is valid (non-zero norm indicates real GT poses available)
    #     # A zero matrix has Frobenius norm ~0, indicating no GT poses for this dataset
    #     fundamental_gt_norm = torch.norm(fundamental_gt.reshape(fundamental_gt.shape[0], -1), dim=1)
    #     has_valid_fundamental_gt = torch.all(fundamental_gt_norm > 1e-6).item()
        
    #     # Select which fundamental matrix to use for epipolar-based metrics
    #     # Use GT fundamental if available, otherwise fall back to predicted
    #     print(fundamental_gt, fundamental_pred)
    #     fundamental_for_epipolar = fundamental_gt if has_valid_fundamental_gt else fundamental_pred
    #     print(fundamental_for_epipolar)
        
    #     # Compute precision, recall, AUCPR
    #     precision, recall, AUCPR = metrics.precision_recall(
    #         source_pixels_matched.detach(),
    #         target_pixels_matched.detach(),
    #         true_pixels_matched.detach() if true_pixels_matched is not None else None,
    #         batch_idx_match.detach(),
    #         scores.detach(),
    #         self.config.MAX_EPIPOLAR_DISTANCE,
    #         fundamental_for_epipolar,
    #     )

    #     # Compute epipolar error using selected fundamental matrix
    #     epipolar_error = metrics.epipolar_error(
    #         source_pixels_matched.cpu(),
    #         target_pixels_matched.cpu(),
    #         # fundamental_for_epipolar.cpu(),   
    #         fundamental_pred.cpu(),
    #         batch_idx_match.cpu(),
    #     )

    #     # Compute fundamental matrix error only if GT is available
    #     # If GT is not available (zeros), set to None to avoid NaN
    #     if has_valid_fundamental_gt:
    #         fundamental_error = metrics.fundamental_error(
    #             fundamental_pred.cpu(), fundamental_gt.cpu()
    #         )[0]
    #     else:
    #         fundamental_error = None
            
    #     # Compute mean matching distance
    #     if true_pixels_matched is not None:
    #         mean_match_distance = metrics.mean_matching_distance(
    #             target_pixels_matched.cpu(),
    #             true_pixels_matched.cpu(),
    #             batch_idx_match.cpu(),
    #         )
    #     else:
    #         mean_match_distance = None

    #     # Return all metrics in a dictionary
    #     return {
    #         "Precision": precision,
    #         "Recall": recall,
    #         "AUCPR": AUCPR,
    #         "EpipolarError": epipolar_error,
    #         "FundamentalError": fundamental_error,
    #         "MDistMean": mean_match_distance,
    #     }

    def match_images(self, image1, image2=None, knn=1):
        """
        Compute correspondences between two images using the unified scoring system.

        Args:
            image1: First image of shape [3, H, W] or [B, 3, H, W]
            image2: Second image of shape [3, H, W] or [B, 3, H, W]. If None, uses the second frame from image1.
            knn: Number of nearest neighbors to consider

        Returns:
            Dictionary containing match information including source/target pixels, scores, etc.
        """
        # Adjust dimensions if missing batch dimension
        if len(image1.shape) == 3:
            image1 = image1.unsqueeze(0)
        if image2 is not None and len(image2.shape) == 3:
            image2 = image2.unsqueeze(0)

        # Create framestack
        if image2 is None:
            framestack = image1
        else:
            framestack = torch.stack([image1, image2], dim=1)

        # Compute model outputs
        with torch.no_grad():
            modeloutput = self.model(framestack)

        # Get initial correspondences with unified scores (without epipolar component)
        correspondence_data = self.compute_correspondences(
            modeloutput, framestack, knn=knn
        )

        # Compute fundamental matrix using RANSAC and update scores with epipolar information
        F, inliers, epipolar_scores = self.RANSAC(
            correspondence_data["source_pixels_matched"],
            correspondence_data["target_pixels_matched"],
            correspondence_data["batch_idx_match"],
        ).values()

        # Update correspondence data with RANSAC results
        correspondence_data.update(
            {
                "F": F,
                "inliers": inliers,
                "epipolar_scores": epipolar_scores,
            }
        )

        # Compute final unified scores with all components
        scores = self.combine_scores(
            correspondence_data["descriptor_scores"],
            correspondence_data["refinement_scores"],
            correspondence_data["epipolar_scores"],
            config=self.config.SCORE_WEIGHTS,
        )

        # Update the final scores
        correspondence_data["scores"] = scores

        return correspondence_data

    def combine_scores(
        self, descriptor_scores, refinement_scores, epipolar_scores, config
    ):
        """
        Combine different scores using a linear combination with configurable weights.

        Args:
            descriptor_scores: tensor of shape (N,) - feature descriptor matching scores
            refinement_scores: tensor of shape (N,) - refinement matching scores
            epipolar_scores: tensor of shape (N,) - epipolar geometry consistency scores
            config: dictionary of weights for each score component

        Returns:
            combined_scores: tensor of shape (N,) - weighted combination of input scores
        """
        # Ensure all inputs are on the same device
        device = descriptor_scores.device

        # Get weights from config
        descriptor_weight = torch.tensor(config.get("DESCRIPTOR", 1.0), device=device)
        refinement_weight = torch.tensor(config.get("REFINEMENT", 1.0), device=device)
        epipolar_weight = torch.tensor(config.get("EPIPOLAR", 1.0), device=device)

        # Normalize weights to sum to 1
        total_weight = descriptor_weight + refinement_weight + epipolar_weight
        descriptor_weight = descriptor_weight / total_weight
        refinement_weight = refinement_weight / total_weight
        epipolar_weight = epipolar_weight / total_weight

        # Vectorized linear combination
        combined_scores = (
            descriptor_weight * descriptor_scores
            + refinement_weight * refinement_scores
            + epipolar_weight * epipolar_scores
        )
        # Check for NaN or non-numeric values in scores
        invalid_mask = torch.isnan(combined_scores) | torch.isinf(combined_scores)
        if invalid_mask.any():
            # Replace invalid scores with zeros
            combined_scores = torch.where(invalid_mask, torch.zeros_like(combined_scores), combined_scores)
        return combined_scores

    def track_points(self, query_points, image1, image2):
        """
        Track arbitrary pixel locations from image1 to image2 without grid quantization.

        Args:
            query_points: [B, Q, 2] or [Q, 2] continuous pixel coordinates (x, y) in image1.
            image1: [B, 3, H, W] or [3, H, W] source frame.
            image2: [B, 3, H, W] or [3, H, W] target frame.

        Returns:
            Dict with:
                tracked_points: [B, Q, 2] continuous pixel coordinates in image2.
                scores:         [B, Q] per-point confidence scores.
        """
        if image1.dim() == 3:
            image1 = image1.unsqueeze(0)
        if image2.dim() == 3:
            image2 = image2.unsqueeze(0)
        if query_points.dim() == 2:
            query_points = query_points.unsqueeze(0)  # [1, Q, 2]

        B, Q, _ = query_points.shape
        framestack = torch.stack([image1, image2], dim=1)  # [B, 2, 3, H, W]

        with torch.no_grad():
            modeloutput = self.model(framestack)

        gated_maps = getattr(self.model, "latest_gated_layer_maps", {})
        src_gated = gated_maps.get("source") if gated_maps else None
        if src_gated is not None and hasattr(self.model, "hierarchical_fusion") and self.model.hierarchical_fusion is not None:
            query_embs = correspondence.sample_gated_embeddings_at_points(
                src_gated["projected_maps"],
                src_gated["gate_weight_maps"],
                query_points,
                self.patch_size,
                self.model.hierarchical_fusion.output_refine,
            )  # [B, Q, C]
        else:
            source_map = embedding2chw(
                modeloutput["source_embedding_match"], embed_dim_last=False
            )  # [B, C, H_p, W_p]
            query_embs = correspondence.sample_embeddings_at_points(
                source_map, query_points, self.patch_size
            )  # [B, Q, C]
        coarse_target = correspondence.query_to_target_coarse(
            query_embs, modeloutput["target_embedding_match"], self.patch_size
        )  # [B, Q, 2]

        refinement_feature_maps = getattr(
            self.model, "latest_refinement_feature_maps", {}
        )
        source_feature_map = refinement_feature_maps.get("source")
        target_feature_map = refinement_feature_maps.get("target")

        use_tracking_head = (
            hasattr(self.model, "tracking_head")
            and self.model.tracking_head is not None
        )

        if use_tracking_head:
            query_fine = correspondence.sample_embeddings_at_points(
                source_feature_map, query_points, self.model.fine_feature_stride
            )  # [B, Q, C_f]
            position_delta, visibility_logit, scores_bq = self.model.tracking_head(
                query_fine, target_feature_map, coarse_target
            )
            tracked = coarse_target + position_delta  # [B, Q, 2]
            return {
                "tracked_points": tracked,
                "scores": scores_bq,
                "visibility_logit": visibility_logit,
                "coarse_target": coarse_target,
            }

        if (
            source_feature_map is not None
            and target_feature_map is not None
            and str(self.config.get("REFINEMENT_METHOD", self.refinement_method)).lower()
            == "feature_softargmax"
        ):
            refinement_window_radius = int(
                self.config.get("FINE_REFINEMENT_WINDOW_RADIUS", 2)
            )
            refinement_stride = int(
                self.config.get(
                    "FINE_FEATURE_STRIDE",
                    getattr(self.model, "fine_feature_stride", 4),
                )
            )
            flat_query = query_points.reshape(B * Q, 2)  # [B*Q, 2]
            flat_coarse = coarse_target.reshape(B * Q, 2)  # [B*Q, 2]
            flat_batch = torch.arange(B, device=query_points.device).repeat_interleave(Q)  # [B*Q]

            _, tgt_offsets, ref_scores = refinement.feature_softargmax_refiner(
                source_feature_map=source_feature_map,
                target_feature_map=target_feature_map,
                source_pixels=flat_query,
                target_pixels=flat_coarse,
                batch_indices=flat_batch,
                window_radius=refinement_window_radius,
                feature_stride=refinement_stride,
                softmax_temperature=float(
                    self.config.get("FINE_REFINEMENT_TEMPERATURE", 0.1)
                ),
            )
            patch_size_virtual = refinement.feature_refinement_patch_size(
                refinement_window_radius, refinement_stride,
            )
            center = patch_size_virtual / 2.0 - 0.5
            tracked = flat_coarse + (tgt_offsets - center)  # [B*Q, 2]
            tracked = tracked.reshape(B, Q, 2)
            ref_scores = ref_scores.reshape(B, Q)
            return {
                "tracked_points": tracked,
                "scores": ref_scores,
                "coarse_target": coarse_target,
            }

        return {
            "tracked_points": coarse_target,
            "scores": torch.ones(B, Q, device=query_points.device),
            "coarse_target": coarse_target,
        }

    def track_points_window(
        self,
        query_points: torch.Tensor,
        frames: torch.Tensor,
        num_refinement_iters: int = 2,
    ):
        """
        Track query points across a temporal window with cached features and
        optional backward-pass refinement.

        Args:
            query_points: [B, Q, 2] initial positions at frame 0.
            frames:       [B, T, 3, H, W] frame window.
            num_refinement_iters: Number of forward-backward refinement sweeps.

        Returns:
            tracks:     [B, Q, T, 2] refined positions.
            visibility: [B, Q, T] visibility logits (if tracking head active, else ones).
        """
        B, T, C_img, H, W = frames.shape
        Q = query_points.shape[1]
        device = query_points.device
        dtype = query_points.dtype

        embeddings_list = []
        fine_maps_list = []
        gated_maps_list = []
        with torch.no_grad():
            for t in range(T):
                (
                    _, matched_tokens, _, _, fine_map, gated_maps_t,
                ) = self.model._encode_image(frames[:, t])
                emb = matched_tokens.permute(0, 2, 1)  # [B, C, N]
                if self.model.resampled_patch_size != 16:
                    emb = chw2embedding(
                        self.model.patchsize_resampler(embedding2chw(emb, False))
                    )
                embeddings_list.append(emb)
                fine_maps_list.append(fine_map)
                gated_maps_list.append(gated_maps_t)

        tracks = torch.zeros(B, Q, T, 2, device=device, dtype=dtype)
        vis_logits = torch.zeros(B, Q, T, device=device, dtype=dtype)
        tracks[:, :, 0, :] = query_points

        use_tracking_head = (
            hasattr(self.model, "tracking_head")
            and self.model.tracking_head is not None
        )

        for _iter in range(num_refinement_iters):
            current_pts = tracks[:, :, 0, :].clone()  # [B, Q, 2]

            has_fusion = (
                hasattr(self.model, "hierarchical_fusion")
                and self.model.hierarchical_fusion is not None
            )
            for t in range(1, T):
                prev_gated = gated_maps_list[t - 1]
                if prev_gated is not None and has_fusion:
                    query_embs = correspondence.sample_gated_embeddings_at_points(
                        prev_gated["projected_maps"],
                        prev_gated["gate_weight_maps"],
                        current_pts,
                        self.patch_size,
                        self.model.hierarchical_fusion.output_refine,
                    )  # [B, Q, C]
                else:
                    source_map = embedding2chw(embeddings_list[t - 1], embed_dim_last=False)  # [B, C, H_p, W_p]
                    query_embs = correspondence.sample_embeddings_at_points(
                        source_map, current_pts, self.patch_size
                    )  # [B, Q, C]
                coarse = correspondence.query_to_target_coarse(
                    query_embs, embeddings_list[t], self.patch_size
                )  # [B, Q, 2]

                src_fine = fine_maps_list[t - 1]
                tgt_fine = fine_maps_list[t]

                if use_tracking_head and src_fine is not None:
                    query_fine = correspondence.sample_embeddings_at_points(
                        src_fine, current_pts, self.model.fine_feature_stride
                    )
                    delta, v_logit, _ = self.model.tracking_head(
                        query_fine, tgt_fine, coarse
                    )
                    tracked = coarse + delta
                    vis_logits[:, :, t] = v_logit.squeeze(-1)
                elif (
                    src_fine is not None
                    and tgt_fine is not None
                    and str(self.config.get("REFINEMENT_METHOD", self.refinement_method)).lower()
                    == "feature_softargmax"
                ):
                    rw = int(self.config.get("FINE_REFINEMENT_WINDOW_RADIUS", 2))
                    rs = int(self.config.get("FINE_FEATURE_STRIDE", getattr(self.model, "fine_feature_stride", 4)))
                    flat_q = current_pts.reshape(B * Q, 2)
                    flat_c = coarse.reshape(B * Q, 2)
                    flat_b = torch.arange(B, device=device).repeat_interleave(Q)
                    _, tgt_off, _ = refinement.feature_softargmax_refiner(
                        source_feature_map=src_fine,
                        target_feature_map=tgt_fine,
                        source_pixels=flat_q,
                        target_pixels=flat_c,
                        batch_indices=flat_b,
                        window_radius=rw,
                        feature_stride=rs,
                        softmax_temperature=float(self.config.get("FINE_REFINEMENT_TEMPERATURE", 0.1)),
                    )
                    ps = refinement.feature_refinement_patch_size(rw, rs)
                    center = ps / 2.0 - 0.5
                    tracked = (flat_c + (tgt_off - center)).reshape(B, Q, 2)
                else:
                    tracked = coarse

                tracks[:, :, t, :] = tracked
                current_pts = tracked

            if _iter < num_refinement_iters - 1 and T > 2:
                for t in range(T - 2, 0, -1):
                    source_map = embedding2chw(embeddings_list[t + 1], embed_dim_last=False)
                    rev_embs = correspondence.sample_embeddings_at_points(
                        source_map, tracks[:, :, t + 1, :], self.patch_size
                    )
                    rev_coarse = correspondence.query_to_target_coarse(
                        rev_embs, embeddings_list[t], self.patch_size
                    )
                    tracks[:, :, t, :] = 0.5 * (tracks[:, :, t, :] + rev_coarse)

        return {
            "tracks": tracks,           # [B, Q, T, 2]
            "visibility": vis_logits,   # [B, Q, T]
        }

    # Wrapper for compute_metrics
    def compute_metrics(self, *args):
        return metrics.compute_metrics(self, *args)

class RegisterGatedHierarchicalFusion(nn.Module):
    """
    Dynamically fuses DINOv3 layers using both local patch evidence and
    global register-token priors.
    """

    def __init__(
        self,
        hidden_dim,
        layer_indices,
        num_register_tokens=4,
        gate_temperature=2.0,
        uniform_mixing=0.1,
        layer_dropout_p=0.1,
        gate_logit_bound=1.5,
    ):
        super().__init__()
        self.layer_indices = list(layer_indices)
        self.num_register_tokens = num_register_tokens
        self.gate_temperature = gate_temperature
        self.uniform_mixing = uniform_mixing
        self.layer_dropout_p = layer_dropout_p
        self.gate_logit_bound = gate_logit_bound
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim, bias=False),
                )
                for _ in self.layer_indices
            ]
        )
        self.local_gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, 1, bias=False),
                )
                for _ in self.layer_indices
            ]
        )
        self.register_gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, 1, bias=False),
                )
                for _ in self.layer_indices
            ]
        )
        for gate_module in list(self.local_gates) + list(self.register_gates):
            nn.init.zeros_(gate_module[-1].weight)
        self.output_refine = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.register_buffer(
            "_layer_index_buf",
            torch.tensor(self.layer_indices, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, hidden_states, return_diagnostics=False):
        projected_layers = []
        gate_logits = []

        for module_idx, layer_idx in enumerate(self.layer_indices):
            layer_hidden = hidden_states[layer_idx]  # [B, N_tokens, C]
            patch_tokens = layer_hidden[:, 1 + self.num_register_tokens :, :]  # [B, N, C]
            register_tokens = layer_hidden[:, 1 : 1 + self.num_register_tokens, :]  # [B, R, C]

            projected = self.projections[module_idx](patch_tokens)  # [B, N, C]
            local_gate = self.local_gates[module_idx](projected)  # [B, N, 1]

            if register_tokens.shape[1] > 0:
                register_context = register_tokens.mean(dim=1)  # [B, C]
            else:
                register_context = layer_hidden[:, 0, :]  # [B, C]
            register_gate = self.register_gates[module_idx](register_context).unsqueeze(1)  # [B, 1, 1]

            projected_layers.append(projected)
            gate_logits.append(
                torch.tanh(local_gate + register_gate) * self.gate_logit_bound
            )

        stacked_projected = torch.stack(projected_layers, dim=0)  # [L, B, N, C]
        stacked_logits = torch.stack(gate_logits, dim=0)  # [L, B, N, 1]
        stacked_logits = stacked_logits - stacked_logits.mean(
            dim=0, keepdim=True
        )  # [L, B, N, 1]
        if self.training and self.layer_dropout_p > 0.0 and len(self.layer_indices) > 1:
            keep_mask = (
                torch.rand(
                    len(self.layer_indices), 1, 1, 1, device=stacked_logits.device
                )
                > self.layer_dropout_p
            )
            if not keep_mask.any():
                keep_mask[torch.randint(len(self.layer_indices), (1,))] = True
            stacked_logits = stacked_logits.masked_fill(~keep_mask, -1e4)
        layer_weights = torch.softmax(
            stacked_logits / max(self.gate_temperature, 1e-6), dim=0
        )  # [L, B, N, 1]
        if self.uniform_mixing > 0.0:
            layer_weights = (
                (1.0 - self.uniform_mixing) * layer_weights
                + self.uniform_mixing / len(self.layer_indices)
            )
        fused = (layer_weights * stacked_projected).sum(dim=0)  # [B, N, C]
        fused = fused + self.output_refine(fused)

        if not return_diagnostics:
            return fused

        layer_weight_maps = layer_weights.squeeze(-1).permute(1, 0, 2).contiguous()  # [B, L, N]
        effective_layer = (
            layer_weight_maps * self._layer_index_buf.view(1, -1, 1)
        ).sum(dim=1)  # [B, N]
        diagnostics = {
            "layer_weights": layer_weight_maps,
            "effective_layer": effective_layer,
            "max_weight": layer_weight_maps.max(dim=1).values,
            "layer_indices": self._layer_index_buf,
            "stacked_projected": stacked_projected,  # [L, B, N, C]
            "layer_weights_LBN1": layer_weights,  # [L, B, N, 1]
        }
        return fused, diagnostics


def _group_norm_groups(num_channels: int) -> int:
    for num_groups in (8, 4, 2, 1):
        if num_channels % num_groups == 0:
            return num_groups
    return 1


class LocalRefinementFeatureHead(nn.Module):
    """
    Builds localizable fine features from RGB appearance and DINO context.
    """

    def __init__(
        self,
        image_channels: int,
        descriptor_dim: int,
        output_dim: int,
        feature_stride: int = 4,
    ):
        super().__init__()
        hidden_dim = max(output_dim // 2, 16)
        self.feature_stride = max(1, int(feature_stride))
        self.rgb_stem = nn.Sequential(
            nn.Conv2d(image_channels, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_norm_groups(hidden_dim), hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_norm_groups(output_dim), output_dim),
            nn.GELU(),
        )
        self.coarse_projection = nn.Sequential(
            nn.Conv2d(descriptor_dim, output_dim, kernel_size=1, bias=False),
            nn.GroupNorm(_group_norm_groups(output_dim), output_dim),
            nn.GELU(),
        )
        self.context_projection = nn.Sequential(
            nn.Conv2d(descriptor_dim, output_dim, kernel_size=1, bias=False),
            nn.GroupNorm(_group_norm_groups(output_dim), output_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(output_dim * 3, output_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_norm_groups(output_dim), output_dim),
            nn.GELU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1, bias=False),
        )

    def forward(
        self,
        image: torch.Tensor,
        coarse_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            image: [B, 3, H, W] RGB image.
            coarse_tokens: [B, C, N] or [B, N, C] descriptor tokens.
            context_tokens: [B, C, N] or [B, N, C] lower-level context tokens.

        Returns:
            [B, C_f, H / s, W / s] normalized fine features.
        """
        target_height = max(image.shape[-2] // self.feature_stride, 1)
        target_width = max(image.shape[-1] // self.feature_stride, 1)
        coarse_map = embedding2chw(coarse_tokens, embed_dim_last=False)
        context_map = embedding2chw(context_tokens, embed_dim_last=False)
        rgb_features = self.rgb_stem(image)  # [B, C_f, H / 4, W / 4]
        if rgb_features.shape[-2:] != (target_height, target_width):
            rgb_features = nn.functional.interpolate(
                rgb_features,
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            )
        coarse_features = nn.functional.interpolate(
            self.coarse_projection(coarse_map),
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )  # [B, C_f, H / s, W / s]
        context_features = nn.functional.interpolate(
            self.context_projection(context_map),
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )  # [B, C_f, H / s, W / s]
        fused = self.fusion(
            torch.cat([rgb_features, coarse_features, context_features], dim=1)
        )  # [B, C_f, H / s, W / s]
        return F.normalize(fused + rgb_features, dim=1)


class TrackingHead(nn.Module):
    """
    Learned local-correlation tracker with visibility prediction.

    Given per-query fine features and a target feature map, computes a local
    correlation window around each current position estimate, applies
    soft-argmax to obtain a sub-pixel position delta, and predicts a
    visibility logit from correlation statistics.
    """

    def __init__(
        self,
        feature_dim: int,
        window_radius: int = 2,
        feature_stride: int = 4,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.window_radius = window_radius
        self.feature_stride = feature_stride
        self.temperature = temperature
        window_width = 2 * window_radius + 1
        window_area = window_width * window_width
        self.visibility_mlp = nn.Sequential(
            nn.Linear(window_area + 2, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        query_features: torch.Tensor,
        target_feature_map: torch.Tensor,
        current_positions: torch.Tensor,
    ):
        """
        Args:
            query_features:     [B, Q, C_f] reference-frame fine features per query.
            target_feature_map: [B, C_f, H_f, W_f] current-frame fine features.
            current_positions:  [B, Q, 2] current estimated pixel positions (x, y).

        Returns:
            position_delta:   [B, Q, 2] sub-pixel offset in image space.
            visibility_logit: [B, Q, 1] raw logit (apply sigmoid for probability).
            scores:           [B, Q] confidence from correlation peak.
        """
        B, Q, C = query_features.shape
        _, _, H_f, W_f = target_feature_map.shape
        device = query_features.device
        dtype = query_features.dtype

        feat_coords = (current_positions.float() + 0.5) / float(self.feature_stride) - 0.5  # [B, Q, 2]

        offsets = torch.arange(
            -self.window_radius, self.window_radius + 1, device=device, dtype=dtype
        )  # [W_w]
        off_y, off_x = torch.meshgrid(offsets, offsets, indexing="ij")
        offset_grid = torch.stack((off_x, off_y), dim=-1)  # [W_w, W_w, 2]

        window_coords = feat_coords.unsqueeze(2).unsqueeze(2) + offset_grid  # [B, Q, W_w, W_w, 2]

        grid = window_coords.clone()
        grid[..., 0] = grid[..., 0] / max(W_f - 1, 1) * 2 - 1
        grid[..., 1] = grid[..., 1] / max(H_f - 1, 1) * 2 - 1

        W_w = 2 * self.window_radius + 1
        grid_flat = grid.reshape(B * Q, W_w, W_w, 2)  # [B*Q, W_w, W_w, 2]
        tgt_expanded = target_feature_map.unsqueeze(1).expand(-1, Q, -1, -1, -1)  # [B, Q, C_f, H_f, W_f]
        tgt_flat = tgt_expanded.reshape(B * Q, C, H_f, W_f)  # [B*Q, C_f, H_f, W_f]
        windows = F.grid_sample(tgt_flat, grid_flat, mode="bilinear", align_corners=True)  # [B*Q, C_f, W_w, W_w]
        windows = F.normalize(windows, dim=1)

        queries_flat = F.normalize(query_features.reshape(B * Q, C), dim=1)  # [B*Q, C]
        logits = torch.einsum("mc,mcij->mij", queries_flat, windows)  # [B*Q, W_w, W_w]
        logits_flat = logits.reshape(B * Q, -1) / max(self.temperature, 1e-6)  # [B*Q, W_w*W_w]
        probs = torch.softmax(logits_flat, dim=1)  # [B*Q, W_w*W_w]
        probs_2d = probs.reshape(B * Q, W_w, W_w)  # [B*Q, W_w, W_w]

        offset_pixels = offset_grid * float(self.feature_stride)  # [W_w, W_w, 2]
        expected_offset = (probs_2d.unsqueeze(-1) * offset_pixels.unsqueeze(0)).sum(dim=(1, 2))  # [B*Q, 2]
        position_delta = expected_offset.reshape(B, Q, 2)  # [B, Q, 2]

        peak_prob = probs.max(dim=1).values  # [B*Q]
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)  # [B*Q]
        max_entropy = torch.log(torch.tensor(float(probs.shape[1]), device=device, dtype=dtype)).clamp_min(1e-6)
        entropy_conf = (1.0 - entropy / max_entropy).clamp(0.0, 1.0)  # [B*Q]
        scores = (0.5 * peak_prob + 0.5 * entropy_conf).clamp(0.0, 1.0).reshape(B, Q)

        delta_mag = position_delta.norm(dim=-1, keepdim=True).reshape(B * Q, 1)  # [B*Q, 1]
        vis_input = torch.cat([probs, peak_prob.unsqueeze(1), delta_mag], dim=1)  # [B*Q, W_w*W_w+2]
        visibility_logit = self.visibility_mlp(vis_input).reshape(B, Q, 1)  # [B, Q, 1]

        return position_delta, visibility_logit, scores


class MatcherModel(FeatureExtractor):
    def __init__(
        self,
        backbone_brand="intel",
        size="beit-base-384",
        resampled_patch_size=8,
        shared_key=None,
        backbone_family="dinov2",
        dino_model_name=None,
        dino_layers="auto",
        fusion_head="register_gated_hierarchical",
        fusion_gate_temperature=2.0,
        fusion_uniform_mixing=0.1,
        fusion_layer_dropout=0.1,
        fusion_gate_logit_bound=1.5,
        refinement_method="fft",
        fine_feature_dim=64,
        fine_feature_stride=4,
        refinement_context_layer="auto",
    ):
        super().__init__(
            backbone_brand=backbone_brand,
            size=size,
            shared_key=shared_key,
            backbone_family=backbone_family,
            dino_model_name=dino_model_name,
            image_size=384,
        )
        self.resampled_patch_size = resampled_patch_size
        self.fusion_head_name = fusion_head
        self.dino_layers = dino_layers
        self.latest_diagnostics = {}
        self.latest_refinement_feature_maps = {}
        self.refinement_method = str(refinement_method).lower()
        self.fine_feature_stride = max(1, int(fine_feature_stride))

        self.depthpredictor = depth_decoding.DPT_Predictor(
            backbone_brand=backbone_brand,
            size=size,
            out_h=384,
            out_w=384,
        )
        self.depth_backbone = self.backbone
        self.depth_backbone_output_key = self.backbone_output_key
        if self.backbone_family == "dinov3":
            self.depth_backbone = getattr(
                backbones, f"DINOv2_{backbone_brand.capitalize()}"
            )(size=size)
            depth_config = self.depth_backbone.model.config.to_dict()
            depth_output_index_key = (
                "out_indices"
                if "swin" in size
                or "beit" in size
                or "facebook" in backbone_brand
                else "backbone_out_indices"
            )
            self.backbone_out_indices = depth_config[depth_output_index_key]
            self.depth_backbone_output_key = (
                "feature_maps" if "swin" in size or "beit" in size else "hidden_states"
            )
            self.layer_indices = self._resolve_dinov3_layers(dino_layers)
            self.hierarchical_fusion = RegisterGatedHierarchicalFusion(
                hidden_dim=self.feature_dim,
                layer_indices=self.layer_indices,
                num_register_tokens=self.num_register_tokens,
                gate_temperature=fusion_gate_temperature,
                uniform_mixing=fusion_uniform_mixing,
                layer_dropout_p=fusion_layer_dropout,
                gate_logit_bound=fusion_gate_logit_bound,
            )
            self.refinement_context_layer = self._resolve_refinement_context_layer(
                refinement_context_layer
            )
        else:
            self.layer_indices = None
            self.hierarchical_fusion = None
            self.refinement_context_layer = None

        self.fine_feature_head = None
        if self.refinement_method == "feature_softargmax":
            self.fine_feature_head = LocalRefinementFeatureHead(
                image_channels=3,
                descriptor_dim=self.feature_dim,
                output_dim=int(fine_feature_dim),
                feature_stride=self.fine_feature_stride,
            )

        self.tracking_head = None
        self._fine_feature_dim = int(fine_feature_dim)

        # Patchsize resampler
        self.patchsize_resampler = lambda x: nn.functional.interpolate(
            x,
            size=(384 // self.resampled_patch_size, 384 // self.resampled_patch_size),
            mode="bilinear",
            align_corners=False,
        )

    def _resolve_dinov3_layers(self, dino_layers):
        num_layers = getattr(self.backbone, "num_hidden_layers", 12)
        if dino_layers is None or str(dino_layers).lower() == "auto":
            n_selected = min(4, num_layers)
            return sorted(
                {
                    int(layer_idx)
                    for layer_idx in torch.linspace(
                        1, num_layers, steps=n_selected
                    ).round().tolist()
                }
            )
        if isinstance(dino_layers, str):
            if dino_layers.lower() == "all":
                return list(range(1, num_layers + 1))
            return [
                max(1, min(num_layers, int(layer.strip())))
                for layer in dino_layers.split(",")
                if layer.strip()
            ]
        return [max(1, min(num_layers, int(layer))) for layer in dino_layers]

    def _resolve_refinement_context_layer(self, refinement_context_layer):
        num_layers = getattr(self.backbone, "num_hidden_layers", 12)
        if (
            refinement_context_layer is None
            or str(refinement_context_layer).lower() == "auto"
        ):
            if self.layer_indices:
                return min(self.layer_indices)
            return max(1, min(num_layers, 2))
        return max(1, min(num_layers, int(refinement_context_layer)))

    def _extract_depth_features(self, image):
        if self.backbone_family == "dinov3":
            depth_features = self.depth_backbone(image)[self.depth_backbone_output_key]
            return list(depth_features) if isinstance(depth_features, tuple) else depth_features
        return self.extract_features(image)

    def _encode_image(self, image):
        from matching.gated_tracking_integration import build_gated_layer_maps

        with torch.no_grad():
            features = self.extract_features(image)

        if self.backbone_family == "dinov3":
            raw_patch_tokens = self.extract_patch_tokens(features[-1])  # [B, N, C]
            matched_patch_tokens, diagnostics = self.hierarchical_fusion(
                features, return_diagnostics=True
            )  # [B, N, C]
            gated_layer_maps = build_gated_layer_maps(diagnostics)
            refinement_context_tokens = self.extract_patch_tokens(
                features[self.refinement_context_layer]
            )  # [B, N, C]
            cls_token = self.extract_cls_token(features[-1])  # [B, C]
            fine_feature_map = self._build_refinement_feature_map(
                image=image,
                matched_patch_tokens=matched_patch_tokens,
                refinement_context_tokens=refinement_context_tokens,
            )
            return (
                raw_patch_tokens,
                matched_patch_tokens,
                cls_token,
                diagnostics,
                fine_feature_map,
                gated_layer_maps,
            )

        matched_tokens = self.lastvitlayer(features[-1])[0]  # [B, N+1, C]
        raw_patch_tokens = self.extract_patch_tokens(features[-1])
        matched_patch_tokens = self.extract_patch_tokens(matched_tokens)
        return (
            raw_patch_tokens,
            matched_patch_tokens,
            self.extract_cls_token(features[-1]),
            None,
            self._build_refinement_feature_map(
                image=image,
                matched_patch_tokens=matched_patch_tokens,
                refinement_context_tokens=raw_patch_tokens,
            ),
            None,
        )

    def _build_refinement_feature_map(
        self,
        image: torch.Tensor,
        matched_patch_tokens: torch.Tensor,
        refinement_context_tokens: torch.Tensor,
    ):
        if self.fine_feature_head is None:
            return None
        return self.fine_feature_head(
            image=image,
            coarse_tokens=matched_patch_tokens.permute(0, 2, 1),
            context_tokens=refinement_context_tokens.permute(0, 2, 1),
        )

    def enable_tracking_head(self, window_radius: int = 2, temperature: float = 0.1):
        """Instantiate the tracking head (idempotent)."""
        if self.tracking_head is not None:
            return
        device = next(self.parameters()).device
        self.tracking_head = TrackingHead(
            feature_dim=self._fine_feature_dim,
            window_radius=window_radius,
            feature_stride=self.fine_feature_stride,
            temperature=temperature,
        ).to(device)

    def save_pretrained_descriptors(self, path: str) -> str:
        """Save only the descriptor-relevant weights (fusion head + fine feature head).

        This produces a lightweight checkpoint suitable for loading into Phase 2
        (tracking) with frozen descriptor layers. The saved dict includes the full
        model state so that ``_load_checkpoint_state_dict`` works without changes,
        but depth predictor and tracking head weights are stripped.

        Args:
            path: Destination ``.pt`` file path.

        Returns:
            The *path* written.
        """
        exclude_prefixes = ("depthpredictor.", "depth_backbone.", "tracking_head.")
        descriptor_state = {
            k: v for k, v in self.state_dict().items()
            if not k.startswith(exclude_prefixes)
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"model_state_dict": descriptor_state}, path)
        logger.info(f"Saved pretrained descriptor checkpoint ({len(descriptor_state)} keys) → {path}")
        return path

    def _load_checkpoint_state_dict(self, state_dict):
        if self.fine_feature_head is None and self.tracking_head is None:
            self.load_state_dict(state_dict)
            return

        incompatible = self.load_state_dict(state_dict, strict=False)
        allowed_missing_prefixes = ("fine_feature_head.", "tracking_head.", "depthpredictor.", "depth_backbone.")
        disallowed_missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith(allowed_missing_prefixes)
        ]
        if disallowed_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Checkpoint is incompatible with the current matcher. "
                f"Missing keys: {disallowed_missing}, unexpected keys: {incompatible.unexpected_keys}"
            )
        if incompatible.missing_keys:
            logger.warning(
                "Checkpoint loaded without fine refinement / tracking head weights; "
                "new branches will start from random initialization."
            )

    def forward(self, framestack):
        """
        Args:
            framestack: torch.Tensor, shape [B, 2, 3, H, W]
        Returns:
            dict with source/target embeddings for matching
        """
        source = framestack[:, 0]  # [B, 3, H, W]
        target = framestack[:, -1] # [B, 3, H, W]
        (
            source_raw_patch_tokens,
            source_matched_patch_tokens,
            source_cls_token,
            source_diagnostics,
            source_fine_feature_map,
            source_gated_maps,
        ) = self._encode_image(source)
        (
            target_raw_patch_tokens,
            target_matched_patch_tokens,
            target_cls_token,
            target_diagnostics,
            target_fine_feature_map,
            target_gated_maps,
        ) = self._encode_image(target)
        self.latest_diagnostics = {
            "source": source_diagnostics,
            "target": target_diagnostics,
        }
        self.latest_refinement_feature_maps = {
            "source": source_fine_feature_map,
            "target": target_fine_feature_map,
        }
        self.latest_gated_layer_maps = {
            "source": source_gated_maps,
            "target": target_gated_maps,
        }

        # Prepare output dict (all shapes [B, C, HW] after permute)
        mono3doutput = {
            "source_embedding": source_raw_patch_tokens.permute(0, 2, 1),  # [B, C, HW]
            "target_embedding": target_raw_patch_tokens.permute(0, 2, 1),  # [B, C, HW]
            "source_embedding_match": source_matched_patch_tokens.permute(0, 2, 1),  # [B, C, HW]
            "target_embedding_match": target_matched_patch_tokens.permute(0, 2, 1),  # [B, C, HW]
            "source_cls": source_cls_token,  # [B, C]
            "target_cls": target_cls_token,  # [B, C]
        }

        # Resample if needed
        if self.resampled_patch_size != 16:
            for key in [
                "source_embedding",
                "target_embedding",
                "source_embedding_match",
                "target_embedding_match",
            ]:
                # [B, C, HW] -> [B, C, h, w] -> resample -> [B, C, h', w'] -> flatten -> [B, C, HW']
                mono3doutput[key] = chw2embedding(
                    self.patchsize_resampler(embedding2chw(mono3doutput[key], False))
                )

        return mono3doutput

    def _compute_depth(self, source_features_dino):
        """Compute depth maps from source features."""
        depthstack = self.depthpredictor(
            source_features_dino,
            self.backbone_out_indices,
        )
        return depthstack.view(-1, 1, depthstack.shape[-2], depthstack.shape[-1])

    def _extract_embeddings(self, features, loftr_shape=True):
        """Extract embeddings for a single image."""
        patch_tokens = self.extract_patch_tokens(features[-1])
        patch_embeddings = embedding2chw(patch_tokens)  # [B, C, H_p, W_p]
        if loftr_shape or self.resampled_patch_size != 16:
            patch_embeddings = self.patchsize_resampler(patch_embeddings)
        return patch_embeddings

    def depth(self, framestack):
        """Inference-only method to return depth maps for one or more images or frame stacks."""
        source_features_dino = self._extract_depth_features(
            framestack[:, 0] if len(framestack.shape) == 5 else framestack
        )
        return self._compute_depth(source_features_dino)

    def embed(self, *frames, mode="chw", loftr_shape=False):
        """Inference-only method to return embeddings for one or more frames."""
        assert mode in ["chw", "seq"], "Invalid mode. Must be 'chw' or 'seq'."
        embs = []
        for image in frames:
            _, matched_patch_tokens, _, _, _, _ = self._encode_image(image)
            emb = embedding2chw(matched_patch_tokens)  # [B, C, H_p, W_p]
            if loftr_shape or self.resampled_patch_size != 16:
                emb = self.patchsize_resampler(emb)
            embs.append(emb)
        embs = [chw2embedding(emb) if mode == "seq" else emb for emb in embs]
        if len(embs) == 1:
            return embs[0]
        return embs

    def fromArtifact(
        self,
        model_name=None,
        local_path=None,
        bucket="alberto-bucket",
        device="cuda",
        pth_namestring=None,
    ):
        """
        Loads a PyTorch model from Google Cloud Storage or local checkpoint

        Args:
            bucket (str): Name of the GCS bucket
            model_name (str): Path to the model file in the bucket
            device (str): Device to load the model on ('cuda' or 'cpu')

        Returns:
            torch.nn.Module: Loaded model, or None if loading failed
        """
        # Try downloading from GCS first
        if pth_namestring is not None:
            if not os.path.exists(pth_namestring):
                raise FileNotFoundError(
                    f"Model checkpoint not found at {pth_namestring}"
                )
            loaded_dict = torch.load(pth_namestring, map_location=device, weights_only=True)
            if "model_state_dict" in loaded_dict.keys():
                self._load_checkpoint_state_dict(loaded_dict["model_state_dict"])
            else:
                self._load_checkpoint_state_dict(loaded_dict)
            return
        if local_path is not None:
            pth_namestring=os.path.join(local_path, f"{model_name}_checkpoint.pth")
            if not os.path.exists(pth_namestring):
                raise FileNotFoundError(
                    f"Model checkpoint not found at {pth_namestring}"
                )
            #  /home/arota/Match/utilities/../runs/offline_20260120_113501/models/offline_20260120_113501_checkpoint.pth
            #  /home/arota/Match/utilities/../runs/offline_20260120_113501/models/offline_20260120_113501_checkpoint.pt
            loaded_dict = torch.load(pth_namestring, map_location=device, weights_only=True)
            self._load_checkpoint_state_dict(loaded_dict["model_state_dict"])

            # Clean up temporary file if it exists
            if os.path.dirname(pth_namestring) == tempfile.gettempdir():
                os.remove(local_path)
        else:
            try:
                local_path = download_from_gcs(model_name=model_name, bucket_name=bucket)
            except Exception:
                # logger.warning(f"Failed to download from GCS: {str(e)}")
                local_path = None

            # If GCS download fails, check local checkpoint directories in order of preference
            if local_path is None:
                # Try multiple possible locations for the checkpoint
                possible_paths = [
                    os.path.join("runs", model_name, "checkpoints", "weights_best.pt"),
                    os.path.join("checkpoints", f"{model_name}_checkpoint.pt"),
                    os.path.join("checkpoints", f"{model_name}.pt"),
                    os.path.join("checkpoints", "weights_best.pt"),  # Fallback
                ]
                
                local_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        local_path = path
                        break
                
                if local_path is None:
                    raise FileNotFoundError(
                        f"Model '{model_name}' not found in GCS bucket {bucket} or locally. "
                        f"Checked paths: {possible_paths}"
                    )
                else:
                    loaded_dict = torch.load(local_path, map_location=device, weights_only=True)
                    self._load_checkpoint_state_dict(loaded_dict["model_state_dict"])
                    