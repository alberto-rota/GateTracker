# -------------------------------------------------------------------------------------------------#

"""Copyright (c) 2024 Asensus Surgical"""

""" Code Developed by: Alberto Rota """
""" Supervision: Uriya Levy, Gal Weizman, Stefano Pomati """

# -------------------------------------------------------------------------------------------------#

from gatetracker.geometry.pipeline import GeometryPipeline
import torch
import torch.nn as nn
import os
from rich import print
import pandas as pd

from gatetracker.utils import *
import gatetracker.utils.engine_init as initialize
from gatetracker.data import collate_fn

import wandb
import numpy as np
from gatetracker.data.augmentation import *

import gatetracker.utils.visualization as viz

from contextlib import contextmanager, nullcontext
import gc
from typing import Union, Dict, Any, Optional, List, Tuple
from gatetracker.utils.logger import get_logger, LogContext
from gatetracker.tracking.losses import compute_pairwise_tracking_losses
from gatetracker.utils.visualization import build_tracking_visualizations
from gatetracker.tracking.tracker import TemporalTracker
from gatetracker.tracking.temporal_refinement import TemporalRefinementNetwork
from gatetracker.metrics.logging import MetricsLogger
from gatetracker.utils.training_phase import (
    build_optimizer_param_groups,
    normalize_pipeline_phase,
    pairwise_tracking_enabled,
)
from gatetracker.utils.formatting import (
    metrics_for_wandb,
    align,
    abbrev_wandb_run_tag,
    abbrev_console_metric_name,
)
from gatetracker.utils.tensor_ops import embedding2chw, chw2embedding
import gatetracker.utils.optimization as optimization


class Engine:
    """
    Main training engine for the MONO3D framework.

    This class orchestrates the entire training process including model initialization,
    data loading, training loops, validation, and testing. It handles all aspects of
    the training pipeline from hyperparameter management to logging and checkpointing.

    The engine supports self-supervised learning with curriculum learning strategies,
    comprehensive logging, early stopping, and integration with Weights & Biases
    for experiment tracking.
    """

    def __init__(
        self,
        model: Union[nn.Module, str, None],
        dataset: Dict[str, Any],
        config: Dict[str, Any],
        notes: str = "",
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Engine object with model, dataset and config.

        Args:
            model: The neural network model to be trained.
            dataset: Dictionary containing 'training', 'validation', and 'test' datasets.
            config: Dictionary containing config like BATCH_SIZE, LEARNING_RATE, etc.
            notes: Additional notes for the training session. Defaults to "".
            **kwargs: Additional keyword arguments.
        """
        self.config = config
        self.config["NOTES"] = notes

        device_dirs = initialize.device_and_directories()
        self.device = device_dirs["device"]
        self.RUNS_DIR = device_dirs["runs_dir"]

        def init(init_func: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
            result = init_func(*args, **kwargs)
            for key, value in result.items():
                setattr(self, key, value)
            return result

        init(initialize.dataloaders, dataset, config)
        init(initialize.dimensions, self.training_dl, config)
        init(initialize.hyperparameters, config)
        init(
            initialize.projections,
            self.height,
            self.width,
            self.device,
            self.learning_rate,
        )
        init(initialize.matching_pipeline, config, model, self.device)
        init(initialize.optimizers, self.matcher.model, config)
        init(initialize.loss_functions, config)
        init(initialize.tracking_loss_functions, config)
        init(initialize.schedulers, self.optimizer, config, self.training_dl)
        init(initialize.transforms, self.height, self.width)
        init(
            initialize.wandb,
            config,
            self.matcher.model,
            notes,
            kwargs.get("no_wandb", False),
        )

        init(initialize.tracking_metrics)
        init(initialize.setup_run_directories, self.RUNS_DIR, self.wandb, False)
        self.config["name"] = self.runname

        geometry_model_name = self.config.get("GEOMETRY_MODEL_NAME", "Ruicheng/moge-2-vits-normal")
        self.geometryPipeline = GeometryPipeline(
            geometry_model_name=geometry_model_name,
            device="cuda",
            height=self.height,
            width=self.width,
            return_normalized_depth=True,
        )
        self.earlystopping = initialize.earlystopping(
            self.earlystopping_patience, self.MODELS_DIR, self.runname
        )

        initialize.save_hyperparameters_json(self.RUN_DIR, self.config)
        self.logger = get_logger(__name__, log_to_file=True, log_dir=self.RUN_DIR)
        self.logger.info(f"Geometry model loaded: {geometry_model_name}", context="ENGINE")

        self.metrics_logger = MetricsLogger(
            wandb_run=self.wandb, run_name=self.runname
        )

        self.refinement_metric_names = {
            "RefinementActiveFraction",
            "RefinementOffsetMean",
            "RefinementScoreMean",
            "CoarseErrorMean",
            "RefinedErrorMean",
            "RefinementGainPx",
            "RefinementGainRatio",
            "RefinementWinRate",
            "RefinementGainConfidenceCorr",
        }

    def _resolve_pretrained_checkpoint_path(self, checkpoint_ref: str) -> Optional[str]:
        """
        Resolve a checkpoint reference into a concrete local file path when possible.

        Supports:
        - Direct checkpoint paths (absolute/relative)
        - Run names (e.g., "wild-feather-787") mapped to known run checkpoint locations
        """
        if not checkpoint_ref:
            return None

        ref = os.path.expanduser(str(checkpoint_ref).strip())
        if os.path.isfile(ref):
            return ref

        looks_like_run_name = (os.sep not in ref) and (not os.path.splitext(ref)[1])
        if not looks_like_run_name:
            return None

        candidate_paths = [
            os.path.join(self.RUNS_DIR, ref, "models", f"{ref}_checkpoint.pth"),
            os.path.join(self.RUNS_DIR, ref, "checkpoints", "weights_best.pt"),
            os.path.join("runs", ref, "models", f"{ref}_checkpoint.pth"),
            os.path.join("runs", ref, "checkpoints", "weights_best.pt"),
            os.path.join("checkpoints", f"{ref}_checkpoint.pth"),
            os.path.join("checkpoints", f"{ref}.pt"),
        ]
        for candidate in candidate_paths:
            if os.path.isfile(candidate):
                return candidate
        return None

    def _load_pretrained_checkpoint(self) -> None:
        """
        Load pretrained weights from config if requested.

        `FROM_PRETRAINED_CHECKPOINT` can be:
        - A checkpoint file path
        - A run name (e.g., "wild-feather-787")
        - A model identifier resolvable by `fromArtifact(model_name=...)`
        """
        checkpoint_ref = self.config.get("FROM_PRETRAINED_CHECKPOINT", "")
        if not checkpoint_ref:
            return

        resolved_path = self._resolve_pretrained_checkpoint_path(checkpoint_ref)
        if resolved_path is not None:
            self.matcher.model.fromArtifact(pth_namestring=resolved_path)
            self.logger.info(
                f"Model loaded from checkpoint: {resolved_path}", context="LOAD"
            )
            return

        self.matcher.model.fromArtifact(model_name=checkpoint_ref)
        self.logger.info(
            f"Model loaded from artifact/model name: {checkpoint_ref}", context="LOAD"
        )

    def _namespace_refinement_metrics(self, metrics_dict, phase_prefix):
        """
        Rewrites refinement metrics to `PHASE/refine/<MetricName>`.

        Args:
            metrics_dict: Flat metric dictionary ready for W&B.
            phase_prefix: Prefix used for this logging call (e.g. `Training`,
                `Validation/epoch`).

        Returns:
            Dict with refinement metrics re-namespaced under `refine/`.
        """
        remapped = {}
        metric_prefix = f"{phase_prefix}/"
        for key, value in metrics_dict.items():
            if key.startswith(metric_prefix):
                metric_name = key[len(metric_prefix):]
                if metric_name in self.refinement_metric_names:
                    remapped[f"{phase_prefix}/refine/{metric_name}"] = value
                    continue
            remapped[key] = value
        return remapped

    def _optimizer_lr_metrics(self) -> Dict[str, float]:
        """Scalar LR keys for logging (primary group + per-group names when present)."""
        out: Dict[str, float] = {
            "HyperParameters/LR": float(self.optimizer.param_groups[0]["lr"]),
        }
        for i, g in enumerate(self.optimizer.param_groups):
            gname = g.get("group_name", f"group_{i}")
            out[f"HyperParameters/LR/{gname}"] = float(g["lr"])
        return out

    def trainloop(self) -> None:
        """
        The main training loop that runs through all epochs, trains the model,
        validates it, and handles early stopping and saving of the model.

        ``config.PHASE`` selects the pipeline:

        - ``pretrain``: Descriptor / matching only (no pairwise tracking).
        - ``end2end``: Descriptors + pairwise tracking losses in ``run_epoch``.
        - ``tracking``: Temporal refinement training (Phase 2) via ``tracking_trainloop``.
        """
        pipeline_phase = normalize_pipeline_phase(self.config)
        if pipeline_phase == "tracking":
            return self.tracking_trainloop()

        self._load_pretrained_checkpoint()
        for e in range(self.epochs):
            self.train()
            training_status = self.validate()

            self.dataset["Training"].reset_sampler()
            self.dataset["Validation"].reset_sampler()

            if e % self.dataset["Training"].max_steps_frameskip == 0 and e > 0:
                self.dataset["Training"].step_frameskip_curriculum()

            self.csv_log_metrics()

            torch.cuda.empty_cache()
            gc.collect()

            if training_status == "EARLYSTOP":
                break

        self.logger.info("TRAINING COMPLETE", context="SAVE")
        self.logger.info(
            f" Run directory: {os.path.abspath(self.RUN_DIR)}", context="SAVE"
        )
        self.logger.info(
            f" Checkpoints  : {os.path.abspath(self.MODELS_DIR)}", context="SAVE"
        )
        self.logger.info(" Metrics      :", context="SAVE")
        self.logger.info(
            f" Training     : {os.path.abspath(os.path.join(self.RUN_DIR, 'training_metrics.csv'))}",
            context="SAVE",
        )
        self.logger.info(
            f" Validation   : {os.path.abspath(os.path.join(self.RUN_DIR, 'validation_metrics.csv'))}",
            context="SAVE",
        )

        images_dir = os.path.join(self.RUN_DIR, "images")
        if os.path.exists(images_dir) and not os.listdir(images_dir):
            try:
                os.rmdir(images_dir)
                self.logger.info(
                    f"Removed unused directory: {images_dir}", context="SAVE"
                )
            except OSError:
                pass

    def train(self) -> str:
        """
        Execute one training epoch.

        Returns:
            Status string indicating training completion.
        """
        return self.run_epoch(phase="Training")

    def validate(self) -> str:
        """
        Execute one validation epoch.

        Returns:
            Status string indicating validation completion or early stopping.
        """
        return self.run_epoch(phase="Validation")

    def test(self) -> None:
        """
        Execute testing on the model.

        When ``config.TRACKING_ONLY`` is True, pairwise matching is skipped and
        only the tracking evaluation is executed.

        When ``config.PHASE == "tracking"``, initializes the TemporalTracker
        and uses it for tracking evaluation instead of pair-wise chaining.
        """
        self._load_pretrained_checkpoint()

        phase = normalize_pipeline_phase(self.config)
        if phase == "tracking":
            self._init_tracking_phase()
            refine_ckpt = os.path.join(self.MODELS_DIR, "tracking_refinement_net.pt")
            if os.path.isfile(refine_ckpt):
                self.temporal_tracker.refinement_net.load_state_dict(
                    torch.load(refine_ckpt, map_location=self.device, weights_only=True)
                )
                self.logger.info(f"Loaded refinement net from {refine_ckpt}", context="TEST")

        tracking_only = bool(self.config.get("TRACKING_ONLY", False))
        if not tracking_only and phase != "tracking":
            self.run_epoch(phase="Test")
        self.run_tracking_evaluation()
        if self.wandb is not None and not tracking_only:
            self.log_tests()

    def run_epoch(self, phase: str = "Training", epoch: Optional[int] = None) -> str:
        """
        Execute a complete epoch for training, validation, or testing.

        Args:
            phase: The phase to execute ('Training', 'Validation', or 'Test').
            epoch: The current epoch index. If None, uses the current step index.

        Returns:
            Status string indicating epoch completion or early stopping.
        """
        assert phase in [
            "Training",
            "Validation",
            "Test",
        ], "Invalid phase. Choose 'Training', 'Validation' or 'Test'."

        PHASE = phase
        TRAINING = PHASE == "Training"
        VALIDATION = PHASE == "Validation"
        TEST = PHASE == "Test"
        if epoch is None:
            epoch = self.step["idx" if TEST else "epoch"]
        if TRAINING:
            self.matcher.model.train()
        else:
            self.matcher.model.eval()
        images_logged = False
        dataset = self.dataset[PHASE]

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.config.WORKERS,
            drop_last=True,
            sampler=dataset.sampler,
            pin_memory=self.config.PIN_MEMORY,
            prefetch_factor=self.config.PREFETCH_FACTOR,
            collate_fn=collate_fn,
        )

        self.switch_optimizer(epoch)
        warmup_base_lrs = [float(g["lr"]) for g in self.optimizer.param_groups]

        if len(dataloader) == 0:
            print("[DATASET]: Empty dataloader, skipping epoch.")
            return "EMPTY"

        with self.choose_if_grad(PHASE):
            for batch_idx, sample in enumerate(dataloader):
                step = epoch * len(dataloader) + batch_idx
                self.step[f"{PHASE}_batch"] += 1

                log_images_this_batch = (
                    batch_idx > 0
                    and batch_idx % self.logfreq_wandb == 0
                    and self.logfreq_wandb > 1
                ) or (batch_idx == len(dataloader) - 1 and not images_logged)
                # Warmup is training-only; validation batches reuse `step` per-epoch so must not
                # trip warmup (would show "W" on val logs and scale LR during eval).
                warming_up = TRAINING and step < self.warmup_steps
                accumulate_gradients = (
                    step >= self.warmup_steps
                    and (batch_idx + 1) % self.gradient_accumulation_steps == 0
                )

                if warming_up:
                    warmup_factor = step / max(self.warmup_steps, 1)
                    for param_group, lr0 in zip(
                        self.optimizer.param_groups, warmup_base_lrs
                    ):
                        param_group["lr"] = lr0 * warmup_factor

                framestack = sample["framestack"].to(self.device)  # [B, T, C, H, W]

                batch_size = framestack.shape[0]

                if sample.get("Ts2t") is not None:
                    camera_pose_gt = sample["Ts2t"].to(self.device)  # [B, 6] or [B, 4, 4]
                else:
                    camera_pose_gt = torch.zeros(batch_size, 6, device=self.device)

                if sample.get("fundamental") is not None:
                    fundamental_gt = sample["fundamental"].to(self.device)
                else:
                    fundamental_gt = torch.zeros(batch_size, 3, 3, device=self.device)

                paths = list(zip(*sample["paths"]))

                if sample.get("intrinsics") is not None:
                    K = sample["intrinsics"].to(self.device)  # [B, 3, 3]
                else:
                    _, _, _, height, width = framestack.shape
                    fx = fy = float(max(height, width))
                    cx, cy = width / 2.0, height / 2.0
                    K = (
                        torch.tensor(
                            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                            dtype=torch.float32,
                            device=self.device,
                        )
                        .unsqueeze(0)
                        .repeat(batch_size, 1, 1)
                    )

                depthstack, _, K_moge = self.geometryPipeline.compute_geometry(
                    framestack
                )
                if sample.get("intrinsics") is not None:
                    K = sample["intrinsics"].to(self.device)  # [B, 3, 3]
                else:
                    K = K_moge[:, 0]

                depthstack = depthstack * self.config.DEPTH_SCALE_FACTOR + self.config.DEPTH_BIAS_FACTOR

                framestack, camera_pose_gt, depthstack = geometric_augmentation(
                    framestack,
                    camera_pose_gt,
                    depthstack,
                    p=(
                        self.config.DATASETS[
                            list(self.config.DATASETS.keys())[0]
                        ].AUGMENTATION_PROBABILITY["GEOMETRIC"]
                        if TRAINING
                        else 0.0
                    ),
                )

                warped = None
                source_matched_points = None
                target_matched_points_true = None
                embedding_mask = None
                embedding_confidence = None
                synthetic_framestack = framestack.clone()
                if TRAINING:
                    gt_output = self.matcher.synthethize_ground_truth(
                        framestack, K, camera_pose_gt, depthstack[:, 0]
                    )
                    warped = gt_output["warped"]
                    source_matched_points = gt_output["source_matched_points"]
                    target_matched_points_true = gt_output["target_matched_points_true"]
                    embedding_mask = gt_output["embedding_mask"]
                    embedding_confidence = gt_output.get("embedding_confidence")
                    synthetic_framestack[:, 1] = warped.clone()
                synthetic_framestack, camera_pose_gt = color_augmentation(
                    synthetic_framestack,
                    camera_pose_gt,
                    p=(
                        self.config.DATASETS[
                            list(self.config.DATASETS.keys())[0]
                        ].AUGMENTATION_PROBABILITY["COLOR"]
                        if TRAINING
                        else 0.0
                    ),
                    target_only=True,
                )
                descriptors = self.matcher.model(synthetic_framestack)

                loss_value = None
                descriptor_loss_value = None
                refinement_loss_value = None
                refinement_active_matches = None
                refinement_weight_mean = None
                triplets, A, P, N = None, None, None, None
                descriptor_pairs_mined = None
                tracking_result = None
                tracking_loss_value = None
                if TRAINING:
                    triplets = self.matcher.mine_triplets(
                        descriptors,
                        source_matched_points,
                        target_matched_points_true,
                        embedding_mask,
                        embedding_confidence
                        if self.config.get("USE_CORRESPONDENCE_CONFIDENCE", True)
                        else None,
                    )
                    A, P, N = [
                        triplets.get(key) for key in ["anchor", "positive", "negative"]
                    ]

                    if getattr(self, "descriptor_loss_name", "triplet") == "infonce":
                        descriptor_loss_tensor = (
                            self.loss_fn(
                                source_to_target_logits=triplets[
                                    "source_to_target_logits"
                                ],
                                source_to_target_positive=triplets[
                                    "source_to_target_positive"
                                ],
                                source_to_target_mask=triplets[
                                    "source_to_target_mask"
                                ],
                                target_to_source_logits=triplets[
                                    "target_to_source_logits"
                                ],
                                target_to_source_positive=triplets[
                                    "target_to_source_positive"
                                ],
                                target_to_source_mask=triplets[
                                    "target_to_source_mask"
                                ],
                                contrastive_weights=triplets[
                                    "contrastive_weights"
                                ],
                            )
                            / self.gradient_accumulation_steps
                        )
                        descriptor_pairs_mined = len(
                            triplets["source_to_target_positive"]
                        )
                    else:
                        descriptor_loss_tensor = (
                            self.loss_fn(A, P, N) / self.gradient_accumulation_steps
                        )
                        descriptor_pairs_mined = len(A)
                descriptor_loss_tensor = None if not TRAINING else descriptor_loss_tensor

                correspondences = self.matcher.compute_correspondences(
                    descriptors,
                    synthetic_framestack,
                    embedding_mask if TRAINING else None,
                )
                source_pixels_matched = correspondences["source_pixels_matched"]
                target_pixels_matched = correspondences["target_pixels_matched"]
                batch_idx_match = correspondences["batch_idx_match"]
                descriptor_scores = correspondences["descriptor_scores"]
                refinement_scores = correspondences["refinement_scores"]
                sim_matrix = correspondences["sim_matrix"]

                refinement_loss_tensor = None

                fundamental_pred, inliers, epipolar_scores = (
                    self.matcher.RANSAC(
                        source_pixels_matched,
                        target_pixels_matched,
                        batch_idx_match,
                    ).values()
                )
                scores = self.matcher.combine_scores(
                    descriptor_scores,
                    refinement_scores,
                    epipolar_scores,
                    self.config.SCORE_WEIGHTS,
                )

                true_pixels_matched = None
                backward_output = {}
                if TRAINING:
                    gt_eval_output = self.matcher.synthethize_ground_truth(
                        synthetic_framestack,
                        K,
                        camera_pose_gt,
                        depthstack[:, 0],
                        source_pixels_matched,
                        batch_idx_match,
                    )
                    warped = gt_eval_output["warped"]
                    source_pixels_matched = gt_eval_output["source_matched_points"]
                    true_pixels_matched = gt_eval_output["target_matched_points_true"]
                    embedding_mask = gt_eval_output["embedding_mask"]
                    refinement_loss_output = (
                        self.matcher.compute_refinement_loss(
                            target_pixels_matched=target_pixels_matched,
                            target_pixels_true=true_pixels_matched,
                            batch_idx_match=batch_idx_match,
                            pixel_confidence=gt_eval_output.get("pixel_confidence"),
                            loss_fn=self.refinement_loss_fn,
                        )
                    )
                    refinement_loss_value = refinement_loss_output["loss"].detach().item()
                    refinement_active_matches = refinement_loss_output["active_matches"]
                    refinement_weight_mean = refinement_loss_output["weight_mean"]
                    refinement_loss_tensor = (
                        refinement_loss_output["loss"]
                        * self.refinement_loss_weight
                        / self.gradient_accumulation_steps
                    )
                    loss_tensor = descriptor_loss_tensor
                    if refinement_loss_tensor is not None:
                        loss_tensor = loss_tensor + refinement_loss_tensor

                    tracking_loss_weight = float(self.config.get("TRACKING_LOSS_WEIGHT", 0.0))
                    if pairwise_tracking_enabled(self.config):
                        tracking_result = compute_pairwise_tracking_losses(
                            matching_pipeline=self.matcher,
                            source_image=synthetic_framestack[:, 0],
                            target_image=synthetic_framestack[:, -1],
                            model_output=descriptors,
                            config=self.config,
                        )
                        tracking_loss_value = tracking_result["loss_total"].detach().item()
                        if tracking_loss_weight > 0:
                            tracking_loss_tensor = (
                                tracking_result["loss_total"]
                                / self.gradient_accumulation_steps
                            )
                            if torch.isfinite(tracking_loss_tensor):
                                loss_tensor = loss_tensor + tracking_loss_tensor

                    backward_output = self.backward_pass(
                        loss_tensor,
                        accumulate_gradients=accumulate_gradients,
                        phase=PHASE,
                    )
                    descriptor_loss_value = descriptor_loss_tensor.detach().item()
                    loss_value = loss_tensor.detach().item()
                if not TRAINING:
                    loss_tensor = None
                    loss_value = None
                    descriptor_loss_tensor = None
                    descriptor_loss_value = None
                    refinement_loss_tensor = None
                    refinement_loss_value = None
                    refinement_active_matches = None
                    refinement_weight_mean = None
                torch.cuda.empty_cache()

                metrics = self.matcher.compute_metrics(
                    source_pixels_matched,
                    target_pixels_matched,
                    true_pixels_matched,
                    batch_idx_match,
                    scores,
                    fundamental_pred,
                    fundamental_gt,
                )
                inlier_count = inliers.count_nonzero().item()
                inlier_percentage = inlier_count / inliers.numel()
                metrics = {
                    k: (v.item() if isinstance(v, torch.Tensor) else v)
                    for k, v in metrics.items()
                }
                architecture_metrics = self.compute_architecture_metrics(
                    descriptors=descriptors,
                    triplets_dict=triplets,
                )
                metrics.update({f"{PHASE}/{k}": v for k, v in architecture_metrics.items()})
                if tracking_result is not None:
                    for key, val in tracking_result["metrics"].items():
                        metrics[f"{PHASE}/tracking/{key}"] = val

                metrics.update(
                    {
                        "Loss": (
                            loss_value
                            if TRAINING
                            else (metrics.get("EpipolarError") if VALIDATION else None)
                        ),
                        "DescriptorLoss": descriptor_loss_value if TRAINING else None,
                        "TrackingLoss": tracking_loss_value if TRAINING else None,
                        "RefinementLoss": refinement_loss_value if TRAINING else None,
                        "NRefinementSupervised": (
                            refinement_active_matches if TRAINING else None
                        ),
                        "RefinementWeightMean": (
                            refinement_weight_mean if TRAINING else None
                        ),
                        "InlierCount": inlier_count,
                        "InlierPercentage": inlier_percentage,
                        "NTripletsMined": (
                            (descriptor_pairs_mined / self.batch_size)
                            if TRAINING
                            else None
                        ),
                        "Gradients/GradNorm": backward_output.get("grad_norm"),
                        "Gradients/WeightNorm": backward_output.get("weight_norm"),
                        **self._optimizer_lr_metrics(),
                        f"Step/{'val' if VALIDATION else ''}batch": self.step[
                            f"{PHASE}_batch"
                        ],
                        f"Step/{'idx' if TEST else 'epoch'}": epoch,
                    }
                )

                images_for_wandb = None
                if log_images_this_batch:
                    images_for_wandb = self.create_all_images(
                        synthetic_framestack,
                        warped if warped is not None else synthetic_framestack[:, 1],
                        sim_matrix,
                        source_pixels_matched,
                        target_pixels_matched,
                        true_pixels_matched,
                        scores,
                        batch_idx_match,
                        triplets,
                        fundamental_pred,
                        descriptors=descriptors,
                        embedding_confidence=embedding_confidence,
                    )
                    if tracking_result is not None:
                        tracking_images = build_tracking_visualizations(
                            source_image=synthetic_framestack[:, 0],
                            target_image=synthetic_framestack[:, -1],
                            tracking_result=tracking_result,
                            model_output=descriptors,
                            matching_pipeline=self.matcher,
                            config=self.config,
                        )
                        if images_for_wandb is None:
                            images_for_wandb = {}
                        for name, img in tracking_images.items():
                            images_for_wandb[f"{PHASE}/tracking/{name}"] = img
                        del tracking_images

                    images_logged = True

                metrics_row = {k: v for k, v in metrics.items()}
                new_row_df = pd.DataFrame([metrics_row])
                self.metrics[PHASE] = pd.concat(
                    [self.metrics[PHASE], new_row_df],
                    ignore_index=True,
                )
                del new_row_df, metrics_row

                self.metrics_logger.log_batch(
                    phase=PHASE,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    total_batches=len(dataloader),
                    metrics=metrics,
                    extra_info="W" if warming_up else None,
                )
                if tracking_result is not None:
                    self.metrics_logger.log_tracking(PHASE, tracking_result["metrics"])

                if images_for_wandb is not None:
                    self.metrics_logger.log_images(PHASE, images_for_wandb)

                if images_for_wandb is not None:
                    del images_for_wandb
                self.log_loaded_paths(paths, PHASE)

                del (
                    framestack,
                    synthetic_framestack,
                    warped,
                    sim_matrix,
                    source_pixels_matched,
                    target_pixels_matched,
                    true_pixels_matched,
                    scores,
                    batch_idx_match,
                    triplets,
                    fundamental_pred,
                    inliers,
                    A,
                    metrics,
                    backward_output,
                    descriptors,
                    camera_pose_gt,
                    depthstack,
                    embedding_mask,
                    descriptor_scores,
                    refinement_scores,
                    epipolar_scores,
                    K,
                    embedding_confidence,
                    tracking_result,
                    tracking_loss_value,
                )
                if "loss_tensor" in locals() and loss_tensor is not None:
                    del loss_tensor
                if "loss_value" in locals():
                    del loss_value
                if "inlier_count" in locals():
                    del inlier_count
                if "inlier_percentage" in locals():
                    del inlier_percentage
                torch.cuda.empty_cache()
                gc.collect()

            self.step[f"{PHASE}_batch"] += 1

        epochstr = (
            "idx" if TEST else "epoch"
        )
        epoch_df_slice = self.metrics[PHASE][
            self.metrics[PHASE][f"Step/{epochstr}"] == epoch
        ]
        epoch_mean_series = epoch_df_slice.mean()
        epoch_mean_dict = epoch_mean_series.to_dict()
        del epoch_df_slice, epoch_mean_series

        epoch_metrics = metrics_for_wandb(epoch_mean_dict, PHASE)
        del epoch_mean_dict

        epoch_metrics = {
            key.replace(PHASE, f"{PHASE}/{epochstr}"): value
            for key, value in epoch_metrics.items()
            if PHASE in key
        }
        epoch_metrics = self._namespace_refinement_metrics(
            epoch_metrics, f"{PHASE}/{epochstr}"
        )

        validation_loss = None
        if VALIDATION:
            loss_key = f"{PHASE}/epoch/Loss"
            if loss_key in epoch_metrics:
                validation_loss = float(epoch_metrics[loss_key])

        if self.wandb is not None:
            self.wandb.log(epoch_metrics)
            del epoch_metrics

        torch.cuda.empty_cache()
        gc.collect()

        max_metrics_rows = 1000
        if len(self.metrics[PHASE]) > max_metrics_rows:
            self.metrics[PHASE] = (
                self.metrics[PHASE].tail(max_metrics_rows).reset_index(drop=True)
            )
            gc.collect()
        if VALIDATION:
            if validation_loss is None:
                epoch_df_slice = self.metrics[PHASE][
                    self.metrics[PHASE][f"Step/{epochstr}"] == epoch
                ]
                if "Loss" in epoch_df_slice.columns and len(epoch_df_slice) > 0:
                    validation_loss = float(epoch_df_slice["Loss"].mean())
                else:
                    raise ValueError(f"Cannot find validation loss for epoch {epoch}")
            self.step[epochstr] += 1
            if self.LRschedulerPlateau is not None:
                self.LRschedulerPlateau.step(validation_loss)
            should_stop = self.earlystopping(
                validation_loss,
                self.matcher.model,
                epoch,
            )
            if should_stop:
                print(">> [EARLYSTOPPING]: Patience Reached, Stopping Training")
                return "EARLYSTOP"
            return "IMPROVED"

        return "COMPLETED"

    @contextmanager
    def choose_if_grad(self, mode: str):
        """
        Conditionally use torch.no_grad based on the given mode.

        Args:
            mode: The current mode ('Training', 'Validation', or 'Test').
        """
        with torch.no_grad() if mode in ["Validation", "Test"] else nullcontext():
            yield

    def backward_pass(
        self,
        loss_tensor: torch.Tensor,
        accumulate_gradients: bool = False,
        phase: str = "Training",
    ) -> Dict[str, Any]:
        """
        Performs the backward pass, including gradient calculation, clipping, and optimization steps.

        Args:
            loss_tensor: The loss tensor to backpropagate.
            accumulate_gradients: If True, will update weights after backpropagation
                                assuming gradient accumulation is complete.
            phase: The current phase ('Training', 'Validation', or 'Test').

        Returns:
            Dictionary containing gradient norms and error status.
        """
        if phase != "Training":
            loss_tensor.detach()
            torch.cuda.empty_cache()

            return {
                "grad_norm": np.nan,
                "weight_norm": np.nan,
            }

        ERROR_IN_BACKWARD_PASS = False
        try:
            loss_tensor.backward()
        except RuntimeError as e:
            print(
                f">> [ERROR]: {e} - Skipping batch {self.step['Training_batch']} in epoch {self.step['epoch']}"
            )
            ERROR_IN_BACKWARD_PASS = True

        grad_norm, weight_norm = optimization.get_norms(
            self.matcher.model.parameters()
        )
        torch.nn.utils.clip_grad_norm_(
            self.matcher.model.parameters(), max_norm=1.0
        )

        if accumulate_gradients:
            if not ERROR_IN_BACKWARD_PASS:
                self.optimizer.step()
                self.optimizer.zero_grad()
            self.LRscheduler.step()

        loss_tensor.detach()
        torch.cuda.empty_cache()

        return {
            "ERROR_IN_BACKWARD_PASS": ERROR_IN_BACKWARD_PASS,
            "grad_norm": grad_norm,
            "weight_norm": weight_norm,
        }

    def switch_optimizer(self, current_epoch: int) -> bool:
        """
        Switches the optimizer if the current epoch matches the switch epoch and
        the bootstrap and refining optimizers are different.

        Args:
            current_epoch: The current epoch number.

        Returns:
            True if the optimizer was switched, False otherwise.
        """
        if (
            current_epoch == self.switch_optimizer_epoch
            and self.optimizer_bootstrap_name != self.optimizer_refining_name
        ):
            print(
                f">> [OPTIMIZER]: "
                f"Switching from [{self.optimizer_bootstrap_name}] to [{self.optimizer_refining_name}]"
            )
            self.in_optswitch_phase = True
            self.optimizer = getattr(optimization, self.optimizer_refining_name)(
                build_optimizer_param_groups(self.matcher.model, self.config),
                lr=float(self.learning_rate),
                weight_decay=float(self.weight_decay),
            )
            return True
        return False

    def run_tracking_evaluation(self) -> None:
        """
        Run point tracking evaluation on all available StereoMIS_Tracking sequences.

        For each sequence:
        - Tracks a dense grid of points and renders a qualitative MP4 video.
        - If GT tracking points exist, computes TAP-Vid metrics (delta_avg, OA, AJ).
        - Logs videos via ``wandb.Video`` and metrics as scalars + summary table.

        Gated on the existence of ``STEREOMIS`` in ``self.config.DATASETS``.
        """
        from gatetracker.data import StereoMISTracking
        from gatetracker.tracking.metrics import compute_tap_metrics
        from gatetracker.utils.visualization import (
            infer_tracks,
            infer_tracks_windowed,
            make_grid_points,
            render_comparison_video,
            render_tracks_video,
        )

        datasets_config = self.config.get("DATASETS", {})
        stereomis_config = datasets_config.get("STEREOMIS", None)
        if stereomis_config is None:
            self.logger.info(
                "No STEREOMIS dataset in config; skipping tracking evaluation.",
                context="TEST",
            )
            return

        config_path = stereomis_config.get("PATH", "StereoMIS_Tracking")
        dataset_rootdir = os.environ.get("DATASET_ROOTDIR")
        if dataset_rootdir and not os.path.isabs(config_path):
            root = os.path.join(dataset_rootdir, config_path)
        else:
            root = config_path

        sequences = StereoMISTracking.available_sequences(root)
        if len(sequences) == 0:
            self.logger.info(
                f"No StereoMIS sequences found under {root}; skipping tracking evaluation.",
                context="TEST",
            )
            return

        height = int(self.config.get("IMAGE_HEIGHT", 384))
        width = int(self.config.get("IMAGE_WIDTH", 384))
        grid_size = int(self.config.get("TRACKING_EVAL_GRID_SIZE", 10))
        use_windowed = bool(self.config.get("TRACKING_EVAL_WINDOWED", False))
        window_size = int(self.config.get("TRACKING_WINDOW_SIZE", 16))
        fps = int(stereomis_config.get("FPS", 4))

        tracking_dir = os.path.join(self.TEST_DIR, "tracking")
        os.makedirs(tracking_dir, exist_ok=True)

        self.logger.info(
            f">> TRACKING EVALUATION on {len(sequences)} StereoMIS sequences "
            f"(windowed={use_windowed}, grid={grid_size}x{grid_size})",
            context="TEST",
        )

        self.matcher.model.eval()
        has_temporal_tracker = hasattr(self, "temporal_tracker") and self.temporal_tracker is not None
        if has_temporal_tracker:
            self.temporal_tracker.refinement_net.eval()
            self.logger.info("Using TemporalTracker for evaluation", context="TEST")

        all_sequence_metrics = []

        for seq_name in sequences:
            self.logger.info(f"  Tracking sequence: {seq_name}", context="TEST")

            try:
                ds = StereoMISTracking(
                    root=root,
                    sequence=seq_name,
                    height=height,
                    width=width,
                )
            except (FileNotFoundError, RuntimeError) as e:
                self.logger.info(
                    f"  Skipping {seq_name}: {e}", context="TEST"
                )
                continue

            if len(ds) < 2:
                self.logger.info(
                    f"  Skipping {seq_name}: fewer than 2 frames.", context="TEST"
                )
                continue

            h, w = ds[0]["image"].shape[1:]
            grid_pts = make_grid_points(h, w, grid_h=grid_size, grid_w=grid_size)

            _cached_all_frames = None
            if has_temporal_tracker:
                _cached_all_frames = torch.stack(
                    [ds[t]["image"] for t in range(len(ds))], dim=0,
                ).unsqueeze(0).to(self.device)  # [1, T, 3, H, W]

            if has_temporal_tracker:
                with torch.no_grad():
                    tracker_out = self.temporal_tracker.track_long_sequence(
                        grid_pts.unsqueeze(0).to(self.device),
                        _cached_all_frames,
                        window_size=window_size,
                    )
                trajectories = tracker_out["tracks"].squeeze(0).permute(1, 0, 2).cpu()  # [T, N, 2]
            elif use_windowed:
                trajectories = infer_tracks_windowed(
                    dataset=ds,
                    matching_pipeline=self.matcher,
                    initial_points=grid_pts,
                    window_size=window_size,
                )
            else:
                trajectories = infer_tracks(
                    dataset=ds,
                    matching_pipeline=self.matcher,
                    initial_points=grid_pts,
                )  # [T, N, 2]

            video_path = os.path.join(tracking_dir, f"{seq_name}_tracking.mp4")
            try:
                render_tracks_video(
                    dataset=ds,
                    trajectories=trajectories,
                    output_path=video_path,
                    fps=max(fps, 4),
                    trail_length=max(5, fps),
                    point_radius=3,
                )
                self.logger.info(
                    f"  Saved tracking video: {video_path}", context="TEST"
                )
            except RuntimeError as e:
                self.logger.info(
                    f"  Video render failed for {seq_name}: {e}", context="TEST"
                )
                video_path = None

            if self.wandb is not None and video_path is not None and os.path.isfile(video_path):
                try:
                    self.wandb.log({
                        f"Test/tracking/{seq_name}_video": wandb.Video(
                            video_path, fps=max(fps, 4), format="mp4"
                        )
                    })
                except Exception as e:
                    self.logger.info(
                        f"  wandb.Video upload failed for {seq_name}: {e}",
                        context="TEST",
                    )

            seq_metrics = {"sequence": seq_name}
            if ds.tracking_points is not None:
                gt_init_pts = ds.tracking_points[:, 0, :]  # [N_pts, 2]

                if has_temporal_tracker:
                    with torch.no_grad():
                        gt_tracker_out = self.temporal_tracker.track_long_sequence(
                            gt_init_pts.unsqueeze(0).to(self.device),
                            _cached_all_frames,
                            window_size=window_size,
                        )
                    eval_traj = gt_tracker_out["tracks"].squeeze(0).permute(1, 0, 2).cpu()  # [T, N_pts, 2]
                elif use_windowed:
                    eval_traj = infer_tracks_windowed(
                        dataset=ds,
                        matching_pipeline=self.matcher,
                        initial_points=gt_init_pts,
                        window_size=window_size,
                    )
                else:
                    eval_traj = infer_tracks(
                        dataset=ds,
                        matching_pipeline=self.matcher,
                        initial_points=gt_init_pts,
                    )  # [T, N_pts, 2]

                gt_tracks = ds.tracking_points  # [N_pts, T_full, 2]
                gt_vis = ds.visibility          # [N_pts, T_full]
                T_pred = eval_traj.shape[0]
                gt_tracks = gt_tracks[:, :T_pred, :]  # [N_pts, T_pred, 2]
                gt_vis = gt_vis[:, :T_pred]            # [N_pts, T_pred]

                pred_tracks = eval_traj.permute(1, 0, 2)       # [N_pts, T_pred, 2]
                pred_vis = torch.ones_like(gt_vis)

                results = compute_tap_metrics(
                    pred_tracks, gt_tracks, gt_vis, pred_vis
                )
                seq_metrics.update(results)

                self.logger.info(
                    f"  {seq_name}: delta_avg={results['delta_avg']:.4f}  "
                    f"OA={results['OA']:.4f}  AJ={results['AJ']:.4f}",
                    context="TEST",
                )

                if self.wandb is not None:
                    self.wandb.log({
                        f"Test/tracking/{seq_name}/delta_avg": results["delta_avg"],
                        f"Test/tracking/{seq_name}/OA": results["OA"],
                        f"Test/tracking/{seq_name}/AJ": results["AJ"],
                    })

                pixel_errors = (pred_tracks - gt_tracks).norm(dim=-1)  # [N_pts, T_pred]

                gt_traj_for_video = gt_tracks.permute(1, 0, 2)  # [T_pred, N_pts, 2]
                cmp_video_path = os.path.join(
                    tracking_dir, f"{seq_name}_gt_vs_pred.mp4"
                )
                try:
                    render_comparison_video(
                        dataset=ds,
                        pred_trajectories=eval_traj,
                        gt_trajectories=gt_traj_for_video,
                        output_path=cmp_video_path,
                        fps=max(fps, 4),
                        trail_length=max(5, fps),
                        point_radius=3,
                        visibility=gt_vis,
                        errors=pixel_errors,
                    )
                    self.logger.info(
                        f"  Saved GT-vs-pred comparison video: {cmp_video_path}",
                        context="TEST",
                    )
                except RuntimeError as e:
                    self.logger.info(
                        f"  Comparison video render failed for {seq_name}: {e}",
                        context="TEST",
                    )
                    cmp_video_path = None

                if (
                    self.wandb is not None
                    and cmp_video_path is not None
                    and os.path.isfile(cmp_video_path)
                ):
                    try:
                        self.wandb.log({
                            f"Test/tracking/{seq_name}_gt_vs_pred": wandb.Video(
                                cmp_video_path, fps=max(fps, 4), format="mp4"
                            )
                        })
                    except Exception as e:
                        self.logger.info(
                            f"  wandb.Video upload failed for comparison {seq_name}: {e}",
                            context="TEST",
                        )
            else:
                self.logger.info(
                    f"  {seq_name}: no GT tracking points, skipping metrics.",
                    context="TEST",
                )

            all_sequence_metrics.append(seq_metrics)

            del ds, trajectories
            if _cached_all_frames is not None:
                del _cached_all_frames
            torch.cuda.empty_cache()

        seqs_with_metrics = [
            m for m in all_sequence_metrics if "delta_avg" in m
        ]
        if seqs_with_metrics:
            mean_delta = np.mean([m["delta_avg"] for m in seqs_with_metrics])
            mean_oa = np.mean([m["OA"] for m in seqs_with_metrics])
            mean_aj = np.mean([m["AJ"] for m in seqs_with_metrics])

            self.logger.info(
                f">> TRACKING MEAN ({len(seqs_with_metrics)} seqs): "
                f"delta_avg={mean_delta:.4f}  OA={mean_oa:.4f}  AJ={mean_aj:.4f}",
                context="TEST",
            )

            if self.wandb is not None:
                self.wandb.log({
                    "Test/tracking/mean_delta_avg": mean_delta,
                    "Test/tracking/mean_OA": mean_oa,
                    "Test/tracking/mean_AJ": mean_aj,
                })

                tracking_table = wandb.Table(
                    columns=["sequence", "delta_avg", "OA", "AJ"],
                    data=[
                        [m["sequence"], m["delta_avg"], m["OA"], m["AJ"]]
                        for m in seqs_with_metrics
                    ],
                )
                self.wandb.log({"Test/tracking/summary": tracking_table})
                del tracking_table

            tracking_csv_path = os.path.join(tracking_dir, "tracking_metrics.csv")
            pd.DataFrame(seqs_with_metrics).to_csv(tracking_csv_path, index=False)
            self.logger.info(
                f"  Tracking metrics CSV: {os.path.abspath(tracking_csv_path)}",
                context="TEST",
            )

    def log_tests(self) -> None:
        """
        Logs the test metrics to Weights and Biases.
        """

        self.logger.info(">> TEST REPORT", context="TEST")
        test_describe = self.metrics["Test"].describe().to_string()
        self.logger.info("\n" + test_describe, context="TEST")
        del test_describe
        self.metrics["Test"].to_csv(self.TEST_DIR + "/test_metrics.csv")
        test_table = wandb.Table(dataframe=self.metrics["Test"])
        self.wandb.log({"Test/Summary": test_table})
        del test_table

        self.logger.info(">> RUN DATA LOCATIONS", context="SAVE")
        self.logger.info(
            f"Run data directory: {os.path.abspath(self.RUN_DIR)}", context="SAVE"
        )
        self.logger.info(
            f"Models saved at: {os.path.abspath(self.MODELS_DIR)}", context="SAVE"
        )
        self.logger.info(f"Metrics CSV files:", context="SAVE")
        self.logger.info(
            f"  - Training: {os.path.abspath(os.path.join(self.RUN_DIR, 'training_metrics.csv'))}",
            context="SAVE",
        )
        self.logger.info(
            f"  - Validation: {os.path.abspath(os.path.join(self.RUN_DIR, 'validation_metrics.csv'))}",
            context="SAVE",
        )
        self.logger.info(
            f"  - Test: {os.path.abspath(os.path.join(self.TEST_DIR, 'test_metrics.csv'))}",
            context="SAVE",
        )

        if hasattr(self.wandb, "url") and self.wandb.url:
            self.logger.info(f"WandB run URL: {self.wandb.url}", context="WANDB")
            project_url = self.wandb.url.rsplit("/", 1)[0]
            self.logger.info(f"WandB project URL: {project_url}", context="WANDB")

        images_dir = os.path.join(self.RUN_DIR, "images")
        if os.path.exists(images_dir) and not os.listdir(images_dir):
            try:
                os.rmdir(images_dir)
                self.logger.info(
                    f"Removed unused directory: {images_dir}", context="SAVE"
                )
            except OSError:
                pass

    def csv_log_metrics(self) -> None:
        """
        Save training and validation metrics to CSV files.
        """

        if self.metrics["Training"] is not None:
            self.metrics["Training"].to_csv(
                os.path.join(self.RUN_DIR, "training_metrics.csv")
            )
        if self.metrics["Validation"] is not None:
            self.metrics["Validation"].to_csv(
                os.path.join(self.RUN_DIR, "validation_metrics.csv")
            )

    def compute_architecture_metrics(
        self,
        descriptors: Dict[str, torch.Tensor],
        triplets_dict: Optional[Dict[str, torch.Tensor]],
    ) -> Dict[str, float]:
        """
        Quantify whether hierarchical fusion and correspondence confidence are useful.
        """
        metrics = {}

        diagnostics = getattr(self.matcher.model, "latest_diagnostics", {})
        source_diag = diagnostics.get("source") if diagnostics is not None else None
        target_diag = diagnostics.get("target") if diagnostics is not None else None

        weight_tensors = []
        if source_diag is not None and source_diag.get("layer_weights") is not None:
            weight_tensors.append(source_diag["layer_weights"])
        if target_diag is not None and target_diag.get("layer_weights") is not None:
            weight_tensors.append(target_diag["layer_weights"])

        if len(weight_tensors) > 0:
            layer_weights = torch.cat(weight_tensors, dim=0)  # [B_all, L, N]
            num_layers = layer_weights.shape[1]
            weights_clamped = layer_weights.clamp_min(1e-8)
            entropy = -(weights_clamped * weights_clamped.log()).sum(dim=1)
            entropy = entropy / max(float(np.log(max(num_layers, 2))), 1e-8)
            metrics["gate/entropy"] = entropy.mean().item()
            metrics["gate/max_weight"] = layer_weights.max(dim=1).values.mean().item()
            split_idx = max(1, num_layers // 2)
            shallow_weights = layer_weights[:, :split_idx]
            deep_weights = (
                layer_weights[:, split_idx:]
                if split_idx < num_layers
                else layer_weights[:, -1:]
            )
            metrics["gate/weight_shallow"] = shallow_weights.mean().item()
            metrics["gate/weight_deep"] = deep_weights.mean().item()
            layer_index_tensors = []
            if source_diag is not None and source_diag.get("layer_weights") is not None:
                layer_index_tensors.append(
                    source_diag["layer_indices"].view(1, -1, 1).expand(
                        source_diag["layer_weights"].shape[0],
                        -1,
                        source_diag["layer_weights"].shape[-1],
                    )
                )
            if target_diag is not None and target_diag.get("layer_weights") is not None:
                layer_index_tensors.append(
                    target_diag["layer_indices"].view(1, -1, 1).expand(
                        target_diag["layer_weights"].shape[0],
                        -1,
                        target_diag["layer_weights"].shape[-1],
                    )
                )
            if len(layer_index_tensors) > 0:
                layer_index_tensor = torch.cat(layer_index_tensors, dim=0)
                effective_layer = (layer_weights * layer_index_tensor).sum(dim=1)
                metrics["gate/effective_layer"] = effective_layer.std(dim=1).mean().item()

        if triplets_dict is None or triplets_dict.get("anchor") is None:
            return metrics

        if len(triplets_dict["anchor"]) == 0:
            return metrics

        fused_pos = nn.functional.cosine_similarity(
            triplets_dict["anchor"], triplets_dict["positive"], dim=1
        )  # [N_triplets]
        fused_neg = nn.functional.cosine_similarity(
            triplets_dict["anchor"], triplets_dict["negative"], dim=1
        )  # [N_triplets]
        fused_margin = fused_pos - fused_neg  # [N_triplets]

        batch_indices = triplets_dict["batch_indices"].long()  # [N_triplets]
        anchor_indices = triplets_dict["anchor_indices"].long()  # [N_triplets]
        positive_indices = triplets_dict["positive_indices"].long()  # [N_triplets]
        negative_indices = triplets_dict["negative_indices"].long()  # [N_triplets]

        source_raw = descriptors["source_embedding"].permute(0, 2, 1)[
            batch_indices, anchor_indices
        ]  # [N_triplets, C]
        target_raw_pos = descriptors["target_embedding"].permute(0, 2, 1)[
            batch_indices, positive_indices
        ]  # [N_triplets, C]
        target_raw_neg = descriptors["target_embedding"].permute(0, 2, 1)[
            batch_indices, negative_indices
        ]  # [N_triplets, C]

        raw_pos = nn.functional.cosine_similarity(source_raw, target_raw_pos, dim=1)
        raw_neg = nn.functional.cosine_similarity(source_raw, target_raw_neg, dim=1)
        raw_margin = raw_pos - raw_neg

        metrics["raw/pos_similarity"] = raw_pos.mean().item()
        metrics["raw/margin"] = raw_margin.mean().item()
        metrics["fused/pos_similarity"] = fused_pos.mean().item()
        metrics["fused/margin"] = fused_margin.mean().item()
        metrics["fused/gain"] = (fused_margin.mean() - raw_margin.mean()).item()

        triplet_confidence = triplets_dict.get("triplet_confidence")
        if triplet_confidence is not None and triplet_confidence.numel() > 0:
            triplet_confidence = triplet_confidence.float()
            metrics["confidence/mean"] = triplet_confidence.mean().item()
            metrics["confidence/std"] = triplet_confidence.std(unbiased=False).item()
            if triplet_confidence.numel() > 1:
                threshold = triplet_confidence.median()
                high_conf = fused_margin[triplet_confidence >= threshold]
                low_conf = fused_margin[triplet_confidence < threshold]
                if high_conf.numel() > 0 and low_conf.numel() > 0:
                    metrics["confidence/high_low_diff"] = (high_conf.mean() - low_conf.mean()).item()

        return metrics

    def _diagnostic_map_to_pil(
        self,
        map_tensor: torch.Tensor,
        label: str,
        resize_to: Tuple[int, int],
        colormap: str = "viridis",
    ):
        return viz.rgb(
            map_tensor,
            as_tensor="pil",
            resize=resize_to,
            interpolation="nearest",
            colormap=colormap,
            label=("top", 24, label),
        )

    def _shared_embedding_rgb(
        self,
        source_embedding: torch.Tensor,
        target_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project source/target embedding maps to RGB with one shared PCA basis.
        """
        # source_embedding / target_embedding: [1, C, H_p, W_p]
        shared_embeddings = torch.cat(
            [source_embedding, target_embedding], dim=0
        )  # [2, C, H_p, W_p]
        _, shared_pca = viz.embedding2color(shared_embeddings, pca=None)
        source_rgb = viz.embedding2color(
            source_embedding, pca=shared_pca
        )  # [1, 3, H_p, W_p]
        target_rgb = viz.embedding2color(
            target_embedding, pca=shared_pca
        )  # [1, 3, H_p, W_p]
        return source_rgb, target_rgb

    def create_architecture_images(
        self,
        framestack: torch.Tensor,
        warped: Optional[torch.Tensor],
        descriptors: Dict[str, torch.Tensor],
        embedding_confidence: Optional[torch.Tensor],
        batch_idx: int = 0,
    ) -> Dict[str, Any]:
        """
        Build visual evidence for the utility of hierarchical fusion and confidence weighting.
        """
        diagnostics = getattr(self.matcher.model, "latest_diagnostics", {})
        source_diag = diagnostics.get("source") if diagnostics is not None else None
        target_diag = diagnostics.get("target") if diagnostics is not None else None

        if source_diag is None or target_diag is None:
            return {}

        image_size = (framestack.shape[-2], framestack.shape[-1])
        grid_size = int(descriptors["source_embedding"].shape[-1] ** 0.5)
        source_raw = embedding2chw(
            descriptors["source_embedding"][batch_idx], embed_dim_last=False
        )  # [1, C, H_p, W_p]
        source_fused = embedding2chw(
            descriptors["source_embedding_match"][batch_idx], embed_dim_last=False
        )  # [1, C, H_p, W_p]
        target_raw = embedding2chw(
            descriptors["target_embedding"][batch_idx], embed_dim_last=False
        )  # [1, C, H_p, W_p]
        target_fused = embedding2chw(
            descriptors["target_embedding_match"][batch_idx], embed_dim_last=False
        )  # [1, C, H_p, W_p]
        source_raw_rgb, target_raw_rgb = self._shared_embedding_rgb(
            source_raw, target_raw
        )
        source_fused_rgb, target_fused_rgb = self._shared_embedding_rgb(
            source_fused, target_fused
        )

        source_overview = viz.panelize(
            viz.rgb(
                framestack[batch_idx, 0], as_tensor="pil", label=("top", 24, "Source")
            ),
            viz.rgb(
                source_raw_rgb,
                as_tensor="pil",
                resize=image_size,
                label=("top", 24, "Raw"),
            ),
            viz.rgb(
                source_fused_rgb,
                as_tensor="pil",
                resize=image_size,
                label=("top", 24, "Fused"),
            ),
            self._diagnostic_map_to_pil(
                source_diag["effective_layer"][batch_idx].view(1, grid_size, grid_size),
                "EffDepth",
                image_size,
                colormap="plasma",
            ),
            self._diagnostic_map_to_pil(
                source_diag["max_weight"][batch_idx].view(
                    1,
                    grid_size,
                    grid_size,
                ),
                "GatePeak",
                image_size,
                colormap="magma",
            ),
            mode="horizontal",
            output_type="pil",
        )

        target_overview = viz.panelize(
            viz.rgb(
                framestack[batch_idx, 1], as_tensor="pil", label=("top", 24, "Target")
            ),
            viz.rgb(
                target_raw_rgb,
                as_tensor="pil",
                resize=image_size,
                label=("top", 24, "Raw"),
            ),
            viz.rgb(
                target_fused_rgb,
                as_tensor="pil",
                resize=image_size,
                label=("top", 24, "Fused"),
            ),
            self._diagnostic_map_to_pil(
                target_diag["effective_layer"][batch_idx].view(1, grid_size, grid_size),
                "EffDepth",
                image_size,
                colormap="plasma",
            ),
            self._diagnostic_map_to_pil(
                target_diag["max_weight"][batch_idx].view(
                    1,
                    grid_size,
                    grid_size,
                ),
                "GatePeak",
                image_size,
                colormap="magma",
            ),
            mode="horizontal",
            output_type="pil",
        )
        fusion_overview = viz.panelize(
            source_overview,
            target_overview,
            mode="vertical",
            output_type="pil",
        )

        source_layer_weights = source_diag["layer_weights"][batch_idx]  # [L, N]
        layer_indices = source_diag["layer_indices"].detach().cpu().tolist()
        num_layers = source_layer_weights.shape[0]
        visualized_layers = (
            torch.linspace(0, num_layers - 1, steps=min(4, num_layers))
            .round()
            .long()
            .tolist()
        )
        weight_panels = [
            viz.rgb(
                framestack[batch_idx, 0],
                as_tensor="pil",
                label=("top", 24, "Source"),
            )
        ]
        for layer_slot in visualized_layers:
            weight_panels.append(
                self._diagnostic_map_to_pil(
                    source_layer_weights[layer_slot].view(1, grid_size, grid_size),
                    f"W{int(layer_indices[layer_slot])}",
                    image_size,
                    colormap="viridis",
                )
            )
        source_weights_panel = viz.panelize(
            *weight_panels, mode="horizontal", output_type="pil"
        )

        confidence_panels = [
            viz.rgb(
                framestack[batch_idx, 1],
                as_tensor="pil",
                label=("top", 24, "Target"),
            ),
            viz.rgb(
                warped[batch_idx] if warped is not None else framestack[batch_idx, 1],
                as_tensor="pil",
                label=("top", 24, "Warped" if warped is not None else "Reference"),
            ),
        ]
        if embedding_confidence is not None:
            confidence_panels.append(
                viz.rgb(
                    embedding_confidence[batch_idx],
                    as_tensor="pil",
                    resize=image_size,
                    interpolation="nearest",
                    colormap="magma",
                    label=("top", 24, "Confidence"),
                )
            )
        confidence_panel = viz.panelize(
            *confidence_panels, mode="horizontal", output_type="pil"
        )

        return {
            "FusionSource": fusion_overview,
            "FusionWeights": source_weights_panel,
            "ConfidenceMap": confidence_panel,
        }

    def console_log_metrics(
        self,
        stage: str,
        epoch: Optional[int] = None,
        batch_idx: Optional[int] = None,
        dataloader_len: Optional[int] = None,
        extra_info: Optional[str] = None,
    ) -> None:
        """
        Print metrics and status information for training, validation, or test.

        Args:
            stage: The current stage ('Training', 'Validation', or 'Test').
            epoch: Current epoch number.
            batch_idx: Current batch index.
            dataloader_len: Length of the dataloader being used.
            extra_info: Additional information to display in the phase indicator.
        """
        epoch_batch_info = (
            f"E{epoch + 1}/{self.epochs} B{batch_idx + 1}/{dataloader_len}"
        )
        # Under Slurm, use scheduler job name (#SBATCH --job-name) verbatim — no abbrev alias.
        slurm_job_name = (os.environ.get("SLURM_JOB_NAME") or "").strip()
        printedrunname = (
            slurm_job_name
            if slurm_job_name
            else abbrev_wandb_run_tag(self.runname)
        )
        tag_col = len(printedrunname) + 1
        metricstring = align(f"{printedrunname}:", tag_col, "right") + epoch_batch_info
        if extra_info is not None:
            metricstring = f"[purple]{extra_info.strip()}[/purple] " + metricstring

        metric_parts: List[str] = []
        if stage in self.metrics.keys():
            if (
                "Loss" in self.metrics[stage].columns
                and self.metrics[stage]["Loss"].iloc[-1] is not None
            ):
                v = float(self.metrics[stage]["Loss"].iloc[-1])
                metric_parts.append(f"[yellow]Loss[/yellow]={v:.4f}")

            for m in self.metrics[stage].columns:
                if (
                    m != "Loss"
                    and "/" not in m
                    and self.metrics[stage][m].iloc[-1] is not None
                ):
                    v = float(self.metrics[stage][m].iloc[-1])
                    lab = abbrev_console_metric_name(m)
                    metric_parts.append(f"[yellow]{lab}[/yellow]={v:.4f}")
        metrs = " ".join(metric_parts)
        self.logger.info(
            metricstring + (" " + metrs if metrs else ""),
            context=stage.upper(),
        )

    def log_loaded_paths(self, paths: List[Tuple[str, str]], phase: str) -> None:
        """
        Log the file paths that were loaded for the current batch.

        Args:
            paths: List of tuples containing (source_path, target_path) pairs.
            phase: The current phase ('Training', 'Validation', or 'Test').
        """
        with open(self.paths_file, mode="a") as file:
            file.write(f"{self.step[f'{phase}_batch']},{paths}\n")

    def create_all_images(
        self,
        framestack: torch.Tensor,
        warped: Optional[torch.Tensor],
        sim_matrix: torch.Tensor,
        source_pixels_matched: torch.Tensor,
        target_pixels_matched: torch.Tensor,
        true_pixels_matched: Optional[torch.Tensor],
        scores: torch.Tensor,
        batch_idx_match: torch.Tensor,
        triplets_dict: Optional[Dict[str, torch.Tensor]],
        fundamental_pred: torch.Tensor,
        descriptors: Optional[Dict[str, torch.Tensor]] = None,
        embedding_confidence: Optional[torch.Tensor] = None,
        patch_size: Optional[int] = None,
        topk: int = 50,
        batch_idx: int = 0,
    ) -> Dict[str, Any]:
        """
        Creates a dictionary of visualization images for training monitoring.

        Args:
            framestack: Tensor containing the source frames.
            warped: Tensor containing the warped (target) frames.
            sim_matrix: Similarity matrix between patches.
            source_pixels_matched: Coordinates of matched pixels in the source image.
            target_pixels_matched: Coordinates of matched pixels in the target image.
            true_pixels_matched: Ground truth coordinates of matches in the target image.
            scores: Confidence scores for the matches.
            batch_idx_match: Batch indices for each match.
            triplets_dict: Dictionary containing triplet information.
            fundamental_pred: Predicted fundamental matrices.
            descriptors: Optional descriptor dictionary for architecture visualizations.
            embedding_confidence: Optional confidence map tensor.
            patch_size: Size of the patches used for matching.
            topk: Number of top matches to visualize.
            batch_idx: Batch index to visualize, defaults to 0.

        Returns:
            Dictionary of PIL image objects for visualization.
        """
        if patch_size is None:
            patch_size = self.patch_size
        batch_filter = batch_idx_match == batch_idx
        if triplets_dict is not None:
            batch_triplet_filter = triplets_dict["batch_indices"] == batch_idx
        TEST = triplets_dict is None
        visualization_dict = {
            "PatchMatches": viz.viewPatchMatches(
                img1=framestack[batch_idx, 0],
                img2=framestack[batch_idx, 1],
                similarity_matrix=sim_matrix[batch_idx],
                patch_size=patch_size,
                topk=topk,
                use_actual_topk=False,
            ),
            "PixelMatches": viz.viewComparePixelMatches(
                img1=framestack[batch_idx, 0],
                img2=framestack[batch_idx, 1],
                pts1=source_pixels_matched[batch_filter],
                pts2=target_pixels_matched[batch_filter],
                pts2_true=(
                    true_pixels_matched[batch_filter]
                    if true_pixels_matched is not None
                    else target_pixels_matched[batch_filter]
                ),
                scores=scores[batch_filter],
                topk=min(topk, len(source_pixels_matched[batch_filter])),
                use_actual_topk=True,
            ),
            "Epipolar": viz.viewEpipolarGeometry(
                img1=framestack[batch_idx, 0],
                img2=framestack[batch_idx, 1],
                pts1=source_pixels_matched[batch_filter],
                pts2=target_pixels_matched[batch_filter],
                scores=scores[batch_filter],
                F=fundamental_pred[batch_idx],
                topk=min(topk, len(source_pixels_matched[batch_filter])),
                use_actual_topk=True,
            ),
            "TripletsMined": (
                viz.viewTriplets(
                    framestack[batch_idx, 0],
                    framestack[batch_idx, 1],
                    anchor_indices=triplets_dict["anchor_indices"][
                        batch_triplet_filter
                    ],
                    positive_indices=triplets_dict["positive_indices"][
                        batch_triplet_filter
                    ],
                    negative_indices=triplets_dict["negative_indices"][
                        batch_triplet_filter
                    ],
                    patch_size=patch_size,
                    num_triplets=min(
                        topk, len(triplets_dict["anchor_indices"][batch_triplet_filter])
                    ),
                )
                if triplets_dict is not None
                else None
            ),
        }

        if descriptors is not None:
            visualization_dict.update(
                self.create_architecture_images(
                    framestack=framestack,
                    warped=warped,
                    descriptors=descriptors,
                    embedding_confidence=embedding_confidence,
                    batch_idx=batch_idx,
                )
            )

        return visualization_dict

    # ==================================================================
    # Phase 2: Temporal Tracking Training
    # ==================================================================

    def _init_tracking_phase(self) -> None:
        """Initialize Phase 2 components: TemporalTracker + sequence dataloader.

        Called once at the start of ``tracking_trainloop``.
        """
        self.temporal_tracker = TemporalTracker.from_config(
            self.matcher, self.config,
        )

        pretrained_ckpt = self.config.get("PRETRAINED_DESCRIPTOR_CKPT", "")
        if pretrained_ckpt:
            resolved = self._resolve_pretrained_checkpoint_path(pretrained_ckpt)
            if resolved:
                self.matcher.model.fromArtifact(pth_namestring=resolved)
                self.logger.info(
                    f"Loaded pretrained descriptors from: {resolved}",
                    context="TRACKING",
                )
            else:
                self.logger.warning(
                    f"Could not resolve pretrained descriptor checkpoint: {pretrained_ckpt}",
                )

        self.temporal_tracker._freeze_matcher_params()
        self.temporal_tracker.refinement_net.to(self.device)

        lr_temporal = float(
            self.config.get(
                "LR_TEMPORAL_REFINEMENT",
                self.config.get("LEARNING_RATE", 1e-4),
            )
        )
        self.tracking_optimizer = torch.optim.Adam(
            self.temporal_tracker.refinement_net.parameters(),
            lr=lr_temporal,
            weight_decay=float(self.config.get("WEIGHT_DECAY", 0)),
        )

        from gatetracker.data import SequenceWindowDataset

        window_size = int(self.config.get("TRACKING_SEQUENCE_LENGTH", 8))
        self._tracking_datasets = {}
        for phase_name in ("Training", "Validation"):
            base_ds = self.dataset[phase_name]
            try:
                seq_ds = SequenceWindowDataset(
                    base_ds,
                    window_size=window_size,
                    stride=max(1, window_size // 2),
                    mode="train" if phase_name == "Training" else "eval",
                )
                self._tracking_datasets[phase_name] = seq_ds
                self.logger.info(
                    f"Built {phase_name} sequence dataset: {len(seq_ds)} windows of {window_size} frames",
                    context="TRACKING",
                )
            except ValueError as e:
                self.logger.warning(f"Could not build {phase_name} sequence dataset: {e}")
                self._tracking_datasets[phase_name] = None

    def tracking_trainloop(self) -> None:
        """Main training loop for Phase 2: temporal tracking."""
        self._load_pretrained_checkpoint()
        self._init_tracking_phase()

        for e in range(self.epochs):
            self._run_tracking_epoch(phase="Training", epoch=e)
            val_status = self._run_tracking_epoch(phase="Validation", epoch=e)

            if hasattr(self.dataset.get("Training", None), "reset_sampler"):
                pass

            self.csv_log_metrics()
            torch.cuda.empty_cache()
            gc.collect()

            if val_status == "EARLYSTOP":
                break

        tracking_ckpt_path = os.path.join(self.MODELS_DIR, "tracking_refinement_net.pt")
        torch.save(
            self.temporal_tracker.refinement_net.state_dict(),
            tracking_ckpt_path,
        )
        self.logger.info(f"Saved tracking refinement network: {tracking_ckpt_path}", context="SAVE")
        self.logger.info("TRACKING TRAINING COMPLETE", context="SAVE")

    def _run_tracking_epoch(self, phase: str = "Training", epoch: int = 0) -> str:
        """Execute one tracking training/validation epoch.

        Args:
            phase: "Training" or "Validation".
            epoch: Current epoch index.

        Returns:
            Status string.
        """
        TRAINING = phase == "Training"
        seq_ds = self._tracking_datasets.get(phase)
        if seq_ds is None:
            # Still run StereoMIS GT / video eval on validation epochs so GT vs pred
            # MP4s are produced even when SequenceWindowDataset(Validation) failed to build.
            if not TRAINING:
                self._tracking_validate_stereomis_gt(epoch)
            return "SKIP"

        if TRAINING:
            self.temporal_tracker.refinement_net.train()
        else:
            self.temporal_tracker.refinement_net.eval()

        def tracking_collate_fn(batch):
            frames = torch.stack([b["frames"] for b in batch], dim=0)  # [B, T, 3, H, W]
            return {"frames": frames}

        tracking_batch_size = int(self.config.get("TRACKING_BATCH_SIZE", max(1, self.batch_size)))
        num_workers = int(self.config.get("TRACKING_WORKERS", max(4, self.config.WORKERS)))
        dataloader = torch.utils.data.DataLoader(
            seq_ds,
            batch_size=tracking_batch_size,
            num_workers=num_workers,
            drop_last=True,
            collate_fn=tracking_collate_fn,
            shuffle=TRAINING,
            pin_memory=True,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )

        if len(dataloader) == 0:
            if not TRAINING:
                self._tracking_validate_stereomis_gt(epoch)
            return "EMPTY"

        num_query_pts = int(self.config.get("TRACKING_NUM_QUERY_POINTS", 64))
        epoch_losses = []

        ctx = nullcontext() if TRAINING else torch.no_grad()
        with ctx:
            for batch_idx, batch in enumerate(dataloader):
                frames = batch["frames"].to(self.device)  # [B, T, 3, H, W]

                loss_dict = self.temporal_tracker.training_step(
                    frames,
                    self.config,
                    num_query_points=num_query_pts,
                    batch=batch,
                    geometry_pipeline=self.geometryPipeline,
                    epoch=epoch,
                )

                loss_total = loss_dict["loss_total"]

                if TRAINING and torch.isfinite(loss_total):
                    loss_total.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.temporal_tracker.refinement_net.parameters(),
                        max_norm=1.0,
                    )
                    self.tracking_optimizer.step()
                    self.tracking_optimizer.zero_grad()

                metrics = {f"{phase}/tracking/{k}": v for k, v in loss_dict["metrics"].items()}
                metrics[f"Step/{'epoch' if phase == 'Training' else 'val_epoch'}"] = epoch
                metrics["Loss"] = loss_total.item()
                m = loss_dict["metrics"]
                metrics["Cycle"] = m.get("loss_cycle", 0.0)
                metrics["Smooth"] = m.get("loss_smooth", 0.0)
                metrics["Desc"] = m.get("loss_desc", 0.0)
                metrics["Feat"] = m.get("loss_feat", 0.0)
                metrics["VisReg"] = m.get("loss_vis", m.get("loss_vis_reg", 0.0))
                metrics["SupPos"] = m.get("loss_sup_pos", 0.0)
                metrics["SupVis"] = m.get("loss_sup_vis", 0.0)
                metrics["PseudoLam"] = m.get("pseudo_lambda", 0.0)
                metrics["PseudoOn"] = m.get("pseudo_gt_active", 0.0)

                new_row = pd.DataFrame([metrics])
                self.metrics[phase] = pd.concat(
                    [self.metrics[phase], new_row], ignore_index=True,
                )
                epoch_losses.append(loss_total.item())

                self.metrics_logger.log_batch(
                    phase=phase,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    total_batches=len(dataloader),
                    metrics=metrics,
                )

                if self.wandb is not None and batch_idx % self.logfreq_wandb == 0:
                    wandb_metrics = {k: v for k, v in metrics.items() if v is not None}
                    self.wandb.log(wandb_metrics)

                del frames, loss_dict, loss_total
                torch.cuda.empty_cache()

        if not TRAINING:
            self._tracking_validate_stereomis_gt(epoch)

        if epoch_losses:
            mean_loss = np.mean(epoch_losses)
            self.logger.info(
                f">> {phase} epoch {epoch+1} mean loss: {mean_loss:.4f}",
                context="TRACKING",
            )
            if self.wandb is not None:
                self.wandb.log({f"{phase}/tracking/epoch_loss": mean_loss})

            if not TRAINING:
                self.step["epoch"] += 1
                if self.LRschedulerPlateau is not None:
                    self.LRschedulerPlateau.step(mean_loss)
                should_stop = self.earlystopping(
                    mean_loss, self.temporal_tracker.refinement_net, epoch,
                )
                if should_stop:
                    self.logger.info(
                        ">> [EARLYSTOPPING]: Patience reached for tracking training",
                        context="TRACKING",
                    )
                    return "EARLYSTOP"

        return "COMPLETED"

    def _tracking_validate_stereomis_gt(self, epoch: int) -> None:
        """Run TAP-Vid metrics on StereoMIS GT tracks for one sequence; log video + scalars to W&B.

        Mirrors ``test_stereomis_p3.py`` / ``run_tracking_evaluation``: grid video plus GT-initialized
        ``track_long_sequence``, comparison video, and ``compute_tap_metrics``.
        """
        if not bool(self.config.get("TRACKING_STEREOMIS_VAL", True)):
            return
        if not hasattr(self, "temporal_tracker") or self.temporal_tracker is None:
            return

        from gatetracker.data import StereoMISTracking
        from gatetracker.tracking.metrics import compute_tap_metrics
        from gatetracker.utils.visualization import (
            make_grid_points,
            render_comparison_video,
            render_tracks_video,
        )

        datasets_config = self.config.get("DATASETS", {}) or {}
        stereomis_config = datasets_config.get("STEREOMIS", None)
        if stereomis_config is None:
            self.logger.info(
                "TRACKING_STEREOMIS_VAL requires DATASETS.STEREOMIS.PATH; skipping.",
                context="TRACKING",
            )
            return

        config_path = stereomis_config.get("PATH", "StereoMIS_Tracking")
        dataset_rootdir = os.environ.get("DATASET_ROOTDIR")
        if dataset_rootdir and not os.path.isabs(config_path):
            root = os.path.join(dataset_rootdir, config_path)
        else:
            root = config_path

        seq_name = str(self.config.get("TRACKING_STEREOMIS_VAL_SEQUENCE", "test_P3_1"))
        seq_dir = os.path.join(root, seq_name)
        if not os.path.isdir(seq_dir):
            available = StereoMISTracking.available_sequences(root)
            if len(available) == 0:
                self.logger.info(
                    f"TRACKING_STEREOMIS_VAL: no sequences under {root}; skipping.",
                    context="TRACKING",
                )
                return
            p3 = [s for s in available if "P3" in s]
            seq_name = p3[0] if p3 else available[0]
            self.logger.info(
                f"TRACKING_STEREOMIS_VAL: resolved sequence {seq_name} (requested path missing).",
                context="TRACKING",
            )

        mf = self.config.get("TRACKING_STEREOMIS_VAL_MAX_FRAMES", None)
        stop: Optional[int] = None
        if mf is not None and mf != "":
            try:
                iv = int(mf)
                if iv > 0:
                    stop = iv
            except (TypeError, ValueError):
                pass

        height = int(self.height)
        width = int(self.width)
        window_size = int(
            self.config.get(
                "TRACKING_SEQUENCE_LENGTH",
                self.config.get("TRACKING_WINDOW_SIZE", 16),
            )
        )
        fps = int(stereomis_config.get("FPS", 4))
        grid_size = int(self.config.get("TRACKING_STEREOMIS_VAL_GRID", 10))

        try:
            ds = StereoMISTracking(
                root=root,
                sequence=seq_name,
                height=height,
                width=width,
                start=0,
                stop=stop,
                step=1,
            )
        except (FileNotFoundError, RuntimeError, OSError) as e:
            self.logger.warning(
                f"TRACKING_STEREOMIS_VAL: could not open {seq_name}: {e}",
                context="TRACKING",
            )
            return

        if len(ds) < 2:
            self.logger.info(
                f"TRACKING_STEREOMIS_VAL: {seq_name} has <2 frames; skipping.",
                context="TRACKING",
            )
            return
        if ds.tracking_points is None:
            self.logger.info(
                f"TRACKING_STEREOMIS_VAL: {seq_name} has no GT (track_pts.pckl); skipping.",
                context="TRACKING",
            )
            return

        out_dir = os.path.join(self.RUN_DIR, "stereomis_val")
        os.makedirs(out_dir, exist_ok=True)
        self.logger.info(
            f"TRACKING_STEREOMIS_VAL: writing videos under {os.path.abspath(out_dir)}",
            context="TRACKING",
        )

        self.temporal_tracker.eval()
        # [1, T, 3, H, W] — same device as training
        all_frames = torch.stack(
            [ds[t]["image"] for t in range(len(ds))], dim=0,
        ).unsqueeze(0).to(self.device)

        h, w = ds[0]["image"].shape[1:]
        grid_pts = make_grid_points(h, w, grid_h=grid_size, grid_w=grid_size)

        prefix = "Validation/stereomis_gt"

        with torch.no_grad():
            grid_out = self.temporal_tracker.track_long_sequence(
                grid_pts.unsqueeze(0).to(self.device),
                all_frames,
                window_size=window_size,
            )
            grid_traj = grid_out["tracks"].squeeze(0).permute(1, 0, 2).cpu()  # [T, N, 2]

            gt_init = ds.tracking_points[:, 0, :].to(self.device)  # [N_gt, 2]
            gt_out = self.temporal_tracker.track_long_sequence(
                gt_init.unsqueeze(0),
                all_frames,
                window_size=window_size,
            )
        pred_tracks = gt_out["tracks"].squeeze(0)  # [N_gt, T, 2]
        vis_logits = gt_out["visibility"].squeeze(0)  # [N_gt, T]
        pred_vis = torch.sigmoid(vis_logits) > 0.5

        T_pred = pred_tracks.shape[1]
        gt_tracks = ds.tracking_points[:, :T_pred, :].to(pred_tracks.device)
        gt_vis = ds.visibility[:, :T_pred].to(pred_tracks.device)

        results = compute_tap_metrics(pred_tracks, gt_tracks, gt_vis, pred_vis)
        self.logger.info(
            f">> {prefix} epoch {epoch + 1} [{seq_name}]  "
            f"delta_avg={results['delta_avg']:.4f}  OA={results['OA']:.4f}  AJ={results['AJ']:.4f}",
            context="TRACKING",
        )

        wb_step = epoch
        log_payload: Dict[str, Any] = {
            f"{prefix}/{seq_name}/delta_avg": results["delta_avg"],
            f"{prefix}/{seq_name}/OA": results["OA"],
            f"{prefix}/{seq_name}/AJ": results["AJ"],
        }

        grid_path = os.path.join(out_dir, f"{seq_name}_grid_ep{epoch:04d}.mp4")
        cmp_path = os.path.join(out_dir, f"{seq_name}_gt_vs_pred_ep{epoch:04d}.mp4")
        try:
            render_tracks_video(
                dataset=ds,
                trajectories=grid_traj,
                output_path=grid_path,
                fps=max(fps, 4),
                trail_length=max(5, fps),
                point_radius=3,
            )
            pixel_errors = (pred_tracks.cpu() - gt_tracks.cpu()).norm(dim=-1)  # [N_gt, T_pred]
            gt_vid = gt_tracks.permute(1, 0, 2).cpu()  # [T, N, 2]
            pred_vid = pred_tracks.permute(1, 0, 2).cpu()
            render_comparison_video(
                dataset=ds,
                pred_trajectories=pred_vid,
                gt_trajectories=gt_vid,
                output_path=cmp_path,
                fps=max(fps, 4),
                trail_length=max(5, fps),
                point_radius=3,
                visibility=gt_vis.cpu(),
                errors=pixel_errors,
            )
            self.logger.info(
                "TRACKING_STEREOMIS_VAL: saved grid tracks video: "
                f"{os.path.abspath(grid_path)}",
                context="TRACKING",
            )
            self.logger.info(
                "TRACKING_STEREOMIS_VAL: saved GT vs predicted tracks video: "
                f"{os.path.abspath(cmp_path)}",
                context="TRACKING",
            )
            if self.wandb is None:
                self.logger.info(
                    "TRACKING_STEREOMIS_VAL: W&B is off; open the MP4 paths above locally.",
                    context="TRACKING",
                )
        except RuntimeError as e:
            self.logger.warning(
                f"TRACKING_STEREOMIS_VAL: video render failed: {e}",
                context="TRACKING",
            )
            grid_path, cmp_path = None, None

        if self.wandb is not None:
            try:
                if grid_path is not None and os.path.isfile(grid_path):
                    log_payload[f"{prefix}/grid_video"] = wandb.Video(
                        grid_path, fps=max(fps, 4), format="mp4"
                    )
                if cmp_path is not None and os.path.isfile(cmp_path):
                    log_payload[f"{prefix}/gt_vs_pred_video"] = wandb.Video(
                        cmp_path, fps=max(fps, 4), format="mp4"
                    )
                self.wandb.log(log_payload, step=wb_step)
            except Exception as e:
                self.logger.warning(
                    f"TRACKING_STEREOMIS_VAL: wandb log failed: {e}",
                    context="TRACKING",
                )

        del all_frames, grid_out, gt_out
        torch.cuda.empty_cache()

    def save_pretrained_descriptors(self, path: Optional[str] = None) -> str:
        """Export a descriptor-only checkpoint for Phase 2 tracking training.

        Args:
            path: Destination file. Defaults to ``<MODELS_DIR>/descriptors_pretrained.pt``.

        Returns:
            The written file path.
        """
        if path is None:
            path = os.path.join(self.MODELS_DIR, "descriptors_pretrained.pt")
        return self.matcher.model.save_pretrained_descriptors(path)

    def reinstantiate_model_from_checkpoint(self) -> None:
        """
        Reinstantiate the model from the latest checkpoint saved.
        """
        self.logger.info(
            f"Attempting reinstatiation from checkpoint @ {self.MODELS_DIR}",
            context="GCLOUD",
        )
        self.matcher.model.fromArtifact(model_name=self.runname, local_path=self.MODELS_DIR)
        self.logger.info(
            f"Model reinstantiated from checkpoint @ {self.runname}",
            context="GCLOUD",
        )
