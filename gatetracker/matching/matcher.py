import torch
import torch.nn as nn
import torch.nn.functional as F

from gatetracker.matching.model import MatcherModel
from gatetracker.matching.helpers import (
    apply_refinement_offsets,
    filter_scores,
    points_to_patches,
)
from gatetracker.matching import (
    learning,
    epipolar,
    refinement,
    correspondence,
    metrics,
)
from gatetracker.geometry import projections

import warnings
from gatetracker.distributed_context import unwrap_model
from gatetracker.utils.logger import get_logger
from gatetracker.utils.training_phase import matcher_should_enable_tracking_head
from gatetracker.utils.tensor_ops import (
    chw2embedding,
    embedding2chw,
    embedding_confidence_from_pixels,
    embedding_mask_from_pixels,
)

logger = get_logger(__name__).set_context("MATCHING")


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
            if isinstance(model, MatcherModel):
                self.model = model.to(self.device)
            elif (
                isinstance(model, str) and model != ""
            ):
                self.model = MatcherModel(
                    shared_key="asdjnasljkn",
                    resampled_patch_size=config["RESAMPLED_PATCH_SIZE"],
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
            ):
                self.model = MatcherModel(
                    resampled_patch_size=config["RESAMPLED_PATCH_SIZE"],
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

        if matcher_should_enable_tracking_head(config):
            self.model.enable_tracking_head(
                window_radius=int(config.get("FINE_REFINEMENT_WINDOW_RADIUS", 2)),
                temperature=float(config.get("FINE_REFINEMENT_TEMPERATURE", 0.1)),
            )

        self.warp = projections.Warp(
            config.IMAGE_HEIGHT,
            config.IMAGE_WIDTH,
        ).to(self.device)
        self.RANSAC_unwrapped = epipolar.FundamentalEstimatorRANSAC()

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
        depthstack,
        source_matched_points=None,
        batch_idx_match=None,
    ):
        if source_matched_points is None:
            source_matched_points = learning.generate_grid(
                num_points=self.config.TRIPLETS_TO_MINE,
                batch_size=framestack.shape[0],
                framestack=framestack,
                device=self.device,
            )

        warping_output = self.warp(
            framestack[:, 0],
            depthstack,
            K,
            camera_pose_gt,
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

        # Patch tokens follow DINO preprocessing (square resize to backbone image_size), not raw
        # config IMAGE_HEIGHT/IMAGE_WIDTH — align confidence maps to that grid before pooling.
        backbone_hw = int(getattr(self.model, "image_size", self.height))
        if pixel_confidence_rgb.shape[-2:] != (backbone_hw, backbone_hw):
            pixel_confidence_rgb = F.interpolate(
                pixel_confidence_rgb,
                size=(backbone_hw, backbone_hw),
                mode="bilinear",
                align_corners=False,
            )

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
        """
        refinement_method = str(
            self.config.get("REFINEMENT_METHOD", self.refinement_method)
        ).lower()
        use_feature_refiner = refinement_method == "feature_softargmax"
        confidence_threshold = self.config.get("PIXEL_MATCHING_SCORE_THRESHOLD", None)
        backbone = unwrap_model(self.model)

        (
            batch_idx_match,
            source_pixels_matched,
            target_pixels_matched,
            source_patches_matched,
            target_patches_matched,
            descriptor_scores,
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
                backbone, "latest_refinement_feature_maps", {}
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
                    getattr(backbone, "fine_feature_stride", 4),
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

        if filter_inliers:
            filtered_points, scores = filter_scores(points_data, descriptor_scores)

            source_pixels_matched = filtered_points["source_pixels"]
            target_pixels_matched = filtered_points["target_pixels"]
            batch_idx_match = filtered_points["batch_idx"]
            source_patches_matched = filtered_points["source_patches"]
            target_patches_matched = filtered_points["target_patches"]
            source_pixel_offset = filtered_points["source_offset"]
            target_pixel_offset = filtered_points["target_offset"]

        coarse_source_pixels = source_pixels_matched.clone()  # [M, 2]
        coarse_target_pixels = target_pixels_matched.clone()  # [M, 2]

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
        return {
            "source_pixels_matched": source_pixels_matched,
            "target_pixels_matched": target_pixels_matched,
            "batch_idx_match": batch_idx_match,
            "descriptor_scores": descriptor_scores,
            "refinement_scores": refinement_scores,
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
        """
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
        epipolar_scores = torch.exp(
            -epipolar_errors / self.config.MAX_EPIPOLAR_DISTANCE
        )

        return {
            "F": F,
            "inliers": inliers,
            "scores": epipolar_scores,
        }

    def match_images(self, image1, image2=None, knn=1):
        """
        Compute correspondences between two images using the unified scoring system.
        """
        if len(image1.shape) == 3:
            image1 = image1.unsqueeze(0)
        if image2 is not None and len(image2.shape) == 3:
            image2 = image2.unsqueeze(0)

        if image2 is None:
            framestack = image1
        else:
            framestack = torch.stack([image1, image2], dim=1)

        with torch.no_grad():
            modeloutput = self.model(framestack)

        correspondence_data = self.compute_correspondences(
            modeloutput, framestack, knn=knn
        )

        F, inliers, epipolar_scores = self.RANSAC(
            correspondence_data["source_pixels_matched"],
            correspondence_data["target_pixels_matched"],
            correspondence_data["batch_idx_match"],
        ).values()

        correspondence_data.update(
            {
                "F": F,
                "inliers": inliers,
                "epipolar_scores": epipolar_scores,
            }
        )

        scores = self.combine_scores(
            correspondence_data["descriptor_scores"],
            correspondence_data["refinement_scores"],
            correspondence_data["epipolar_scores"],
            config=self.config.SCORE_WEIGHTS,
        )

        correspondence_data["scores"] = scores

        return correspondence_data

    def combine_scores(
        self, descriptor_scores, refinement_scores, epipolar_scores, config
    ):
        """
        Combine different scores using a linear combination with configurable weights.
        """
        device = descriptor_scores.device

        descriptor_weight = torch.tensor(config.get("DESCRIPTOR", 1.0), device=device)
        refinement_weight = torch.tensor(config.get("REFINEMENT", 1.0), device=device)
        epipolar_weight = torch.tensor(config.get("EPIPOLAR", 1.0), device=device)

        total_weight = descriptor_weight + refinement_weight + epipolar_weight
        descriptor_weight = descriptor_weight / total_weight
        refinement_weight = refinement_weight / total_weight
        epipolar_weight = epipolar_weight / total_weight

        combined_scores = (
            descriptor_weight * descriptor_scores
            + refinement_weight * refinement_scores
            + epipolar_weight * epipolar_scores
        )
        invalid_mask = torch.isnan(combined_scores) | torch.isinf(combined_scores)
        if invalid_mask.any():
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
        backbone = unwrap_model(self.model)

        with torch.no_grad():
            modeloutput = self.model(framestack)

        gated_maps = getattr(backbone, "latest_gated_layer_maps", {})
        src_gated = gated_maps.get("source") if gated_maps else None
        if src_gated is not None and hasattr(backbone, "hierarchical_fusion") and backbone.hierarchical_fusion is not None:
            query_embs = correspondence.sample_gated_embeddings_at_points(
                src_gated["projected_maps"],
                src_gated["gate_weight_maps"],
                query_points,
                self.patch_size,
                backbone.hierarchical_fusion.output_refine,
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
            backbone, "latest_refinement_feature_maps", {}
        )
        source_feature_map = refinement_feature_maps.get("source")
        target_feature_map = refinement_feature_maps.get("target")

        use_tracking_head = (
            hasattr(backbone, "tracking_head")
            and backbone.tracking_head is not None
        )

        if use_tracking_head:
            query_fine = correspondence.sample_embeddings_at_points(
                source_feature_map, query_points, backbone.fine_feature_stride
            )  # [B, Q, C_f]
            position_delta, visibility_logit, scores_bq = backbone.tracking_head(
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
                    getattr(backbone, "fine_feature_stride", 4),
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
        backbone = unwrap_model(self.model)
        with torch.no_grad():
            for t in range(T):
                (
                    _, matched_tokens, _, _, fine_map, gated_maps_t,
                ) = backbone._encode_image(frames[:, t])
                emb = matched_tokens.permute(0, 2, 1)  # [B, C, N]
                if backbone.resampled_patch_size != 16:
                    emb = chw2embedding(
                        backbone.patchsize_resampler(embedding2chw(emb, False))
                    )
                embeddings_list.append(emb)
                fine_maps_list.append(fine_map)
                gated_maps_list.append(gated_maps_t)

        tracks = torch.zeros(B, Q, T, 2, device=device, dtype=dtype)
        vis_logits = torch.zeros(B, Q, T, device=device, dtype=dtype)
        tracks[:, :, 0, :] = query_points

        use_tracking_head = (
            hasattr(backbone, "tracking_head")
            and backbone.tracking_head is not None
        )

        for _iter in range(num_refinement_iters):
            current_pts = tracks[:, :, 0, :].clone()  # [B, Q, 2]

            has_fusion = (
                hasattr(backbone, "hierarchical_fusion")
                and backbone.hierarchical_fusion is not None
            )
            for t in range(1, T):
                prev_gated = gated_maps_list[t - 1]
                if prev_gated is not None and has_fusion:
                    query_embs = correspondence.sample_gated_embeddings_at_points(
                        prev_gated["projected_maps"],
                        prev_gated["gate_weight_maps"],
                        current_pts,
                        self.patch_size,
                        backbone.hierarchical_fusion.output_refine,
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
                        src_fine, current_pts, backbone.fine_feature_stride
                    )
                    delta, v_logit, _ = backbone.tracking_head(
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
                    rs = int(self.config.get("FINE_FEATURE_STRIDE", getattr(backbone, "fine_feature_stride", 4)))
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

    def compute_metrics(self, *args):
        return metrics.compute_metrics(self, *args)
