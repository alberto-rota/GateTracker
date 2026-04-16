# -------------------------------------------------------------------------------------------------#

"""Copyright (c) 2024 Asensus Surgical"""

""" Code Developed by: Alberto Rota """
""" Supervision: Uriya Levy, Gal Weizman, Stefano Pomati """

# -------------------------------------------------------------------------------------------------#

from readline import redisplay
from utilities.geometry_pipeline import GeometryPipeline
import torch
import torch.nn as nn
import os
from rich import print
import pandas as pd

from utilities import *
import utilities.engine_initializers as initialize
from dataset.loader import collate_fn

import os
import wandb
import numpy as np
import augmentation as aug

# import matching as match
import utilities.visualization as viz

from contextlib import contextmanager, nullcontext
import gc
from typing import Union, Dict, Any, Optional, List, Tuple
from logger import get_logger, LogContext
from matching.tracking_losses import compute_tracking_losses
from utilities.tracking_visualization import build_tracking_visualizations
from matching.tracker import TemporalTracker
from matching.temporal_refinement import TemporalRefinementNetwork


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
        torch.autograd.set_detect_anomaly(True)
        # Store the model and config
        self.config = config
        self.config["NOTES"] = notes

        # Initialize device and directories
        device_dirs = initialize.device_and_directories()
        self.device = device_dirs["device"]
        self.RUNS_DIR = device_dirs["runs_dir"]

        # Function to set attributes from initialization functions
        def init(init_func: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
            result = init_func(*args, **kwargs)
            for key, value in result.items():
                setattr(self, key, value)
            return result

        # Initialize all components
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
        # Initialize early stopping
        self.earlystopping = initialize.earlystopping(
            self.earlystopping_patience, self.MODELS_DIR, self.runname
        )

        # Save hyperparameters to json
        initialize.save_hyperparameters_json(self.RUN_DIR, self.config)
        self.logger = get_logger(__name__, log_to_file=True, log_dir=self.RUN_DIR)
        self.logger.info(f"Geometry model loaded: {geometry_model_name}", context="ENGINE")

        # Log run information at initialization time
        # self.logger.info(f"Run initialized: {self.runname}", context="INIT")
        # if hasattr(self.wandb, "url") and self.wandb.url:
        #     self.logger.info(f"WandB run URL: {self.wandb.url}", context="WANDB")
        #     project_url = self.wandb.url.rsplit("/", 1)[0]
        #     self.logger.info(f"WandB project URL: {project_url}", context="WANDB")

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

        # Interpret bare names as run names and probe common checkpoint locations.
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

        # Fall back to artifact/model-name resolution logic.
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
                metric_name = key[len(metric_prefix) :]
                if metric_name in self.refinement_metric_names:
                    remapped[f"{phase_prefix}/refine/{metric_name}"] = value
                    continue
            remapped[key] = value
        return remapped

    def trainloop(self) -> None:
        """
        The main training loop that runs through all epochs, trains the model,
        validates it, and handles early stopping and saving of the model.

        This method orchestrates the complete training process including:
        - Epoch iteration with training and validation
        - Curriculum learning progression
        - Early stopping based on validation loss
        - Comprehensive logging and checkpointing
        - Final training summary and cleanup

        When ``config.PHASE == "tracking"``, delegates to the Phase 2 temporal
        tracking training loop instead.
        """
        phase = str(self.config.get("PHASE", "pretrain")).lower()
        if phase == "tracking":
            return self.tracking_trainloop()

        self._load_pretrained_checkpoint()
        for e in range(self.epochs):
            ### TRAINING + VALIDATION FOR EACH EPOCH
            self.train()  # Train the model for one epoch
            training_status = self.validate()  # Train the model for one epoch

            ### RESET SAMPLERS FOR INCREASED ROBUSTNESS
            self.dataset["Training"].reset_sampler()
            self.dataset["Validation"].reset_sampler()

            # Step the curriculum learning
            if e % self.dataset["Training"].max_steps_frameskip == 0 and e > 0:
                self.dataset["Training"].step_frameskip_curriculum()

            self.csv_log_metrics()

            # Periodic memory cleanup every epoch to prevent accumulation
            torch.cuda.empty_cache()
            gc.collect()

            ### BREAK IF EARLYSTOP
            if training_status == "EARLYSTOP":
                break  # Exit the training loop if early stopping condition is met

        # Log locations of important data at the end of training
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

        # Log WandB URLs again for convenience
        # if hasattr(self.wandb, "url") and self.wandb.url:
        #     self.logger.info(f"WandB run URL: {self.wandb.url}", context="WANDB")
        #     project_url = self.wandb.url.rsplit("/", 1)[0]
        #     self.logger.info(f"WandB project URL: {project_url}", context="WANDB")

        # Remove unused IMAGES_DIR if it exists and is empty
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

        This method runs the model in test mode and logs comprehensive test metrics
        including performance statistics and visualization results. After pairwise
        matching evaluation, runs point tracking evaluation on all available
        StereoMIS_Tracking sequences (metrics + qualitative videos).

        When ``config.TRACKING_ONLY`` is True, pairwise matching is skipped and
        only the tracking evaluation is executed.

        When ``config.PHASE == "tracking"``, initializes the TemporalTracker
        and uses it for tracking evaluation instead of pair-wise chaining.
        """
        self._load_pretrained_checkpoint()

        phase = str(self.config.get("PHASE", "pretrain")).lower()
        if phase == "tracking":
            self._init_tracking_phase()
            # Load refinement net weights if available
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

        This method handles the complete forward pass, loss computation, backward pass,
        and metric logging for a single epoch. It supports different phases (training,
        validation, test) with appropriate behavior for each.

        Args:
            phase: The phase to execute ('Training', 'Validation', or 'Test').
            epoch: The current epoch index. If None, uses the current step index.

        Returns:
            Status string indicating epoch completion or early stopping.

        Raises:
            AssertionError: If phase is not one of the valid options.
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
        base_lr = self.optimizer.param_groups[0]["lr"]

        if len(dataloader) == 0:
            print("[DATASET]: Empty dataloader, skipping epoch.")
            return "EMPTY"

        # Base learning rate for warmup
        with self.choose_if_grad(PHASE):
            for batch_idx, sample in enumerate(dataloader):
                ### INITIALIZATION
                step = epoch * len(dataloader) + batch_idx
                self.step[f"{PHASE}_batch"] += 1

                ### Batch Flags
                # True if we should log images on this batch
                log_images_this_batch = (
                    batch_idx > 0
                    and batch_idx % self.logfreq_wandb == 0
                    and self.logfreq_wandb > 1
                ) or (batch_idx == len(dataloader) - 1 and not images_logged)
                # True if we should scale the learning rate for the warmup phase
                warming_up = step < self.warmup_steps
                # True if we should accumulate gradients, false if grads should be backpropagated
                accumulate_gradients = (
                    step >= self.warmup_steps
                    and (batch_idx + 1) % self.gradient_accumulation_steps == 0
                )

                # Warmup logic
                if warming_up:
                    warmup_factor = step / self.warmup_steps
                    current_lr = base_lr * warmup_factor
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = current_lr

                ### ACCESSING SAMPLE ELEMENTS
                framestack = sample["framestack"].to(self.device)  # [B, T, C, H, W]

                # Handle None values for poses and intrinsics
                batch_size = framestack.shape[0]

                if sample.get("Ts2t") is not None:
                    camera_pose_gt = sample["Ts2t"].to(
                        self.device
                    )  # [B, 6] or [B, 4, 4]
                else:
                    # Create dummy pose if not available (should not happen if random_pose is enabled)
                    # Check shape of Ts2t from other samples or use default (Euler format: 6 values)
                    # Try to infer format from first non-None value in batch, or default to Euler
                    camera_pose_gt = torch.zeros(batch_size, 6, device=self.device)

                if sample.get("fundamental") is not None:
                    fundamental_gt = sample["fundamental"].to(self.device)
                else:
                    # Create dummy fundamental matrix if not available
                    fundamental_gt = torch.zeros(batch_size, 3, 3, device=self.device)

                paths = list(
                    zip(*sample["paths"])
                )  # list of tuples (source_path, target_path)

                if sample.get("intrinsics") is not None:
                    K = sample["intrinsics"].to(self.device)  # [B, 3, 3]
                else:
                    # Create default intrinsics matrix based on image dimensions
                    # framestack shape: [B, T, C, H, W]
                    _, _, _, height, width = framestack.shape
                    # Default intrinsics: focal length ~= image dimension, principal point at center
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

                ### DEPTHMAP + INTRINSIC INFERENCE - From MoGe-2
                depthstack, _, K_moge = self.geometryPipeline.compute_geometry(
                    framestack
                )
                if sample.get("intrinsics") is not None:
                    K = sample["intrinsics"].to(self.device)  # [B, 3, 3]
                else:
                    K = K_moge[:, 0]

                ### RESCALING DEPTHSTACK TO MEANINGFUL RANGE (WRT FOCAL LENGTH AND CAMERA MOTION)
                depthstack = depthstack * self.config.DEPTH_SCALE_FACTOR + self.config.DEPTH_BIAS_FACTOR

                ### GEOMETRIC AUGMENTATION
                framestack, camera_pose_gt, depthstack = aug.geometric_augmentation(
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
                    # target_only=True,
                )
                # tprint(K)
                # display(channels(depthstack))

                ### SYNTHETIC SUPERVISION (TRAINING ONLY)
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
                ### AUGMENTATIONS
                synthetic_framestack, camera_pose_gt = aug.color_augmentation(
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
                ### BACKBONE FORWARD PASS
                descriptors = self.matcher.model(synthetic_framestack)

                ### MINE TRIPLETS
                loss_value = None  # Initialize loss_value
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

                ### FINDING PIXEL CORRESPONDENCES
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

                ### FUNDAMENTAL MATRIX ESTIMATION
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

                ### RETRIEVING PSEUDO-GROUND TRUTH (TRAINING ONLY) FOR SUPERVISED LOSSES
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
                    if self.config.get("TRACKING_MODE", False):
                        tracking_result = compute_tracking_losses(
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
                # These variables are not calculated outside training
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

                ### METRICS COMPUTATION
                metrics = self.matcher.compute_metrics(
                    source_pixels_matched,
                    target_pixels_matched,
                    true_pixels_matched,
                    batch_idx_match,
                    scores,
                    fundamental_pred,
                    fundamental_gt,
                )
                # Compute inlier stats once to avoid duplicate GPU operations
                inlier_count = inliers.count_nonzero().item()
                inlier_percentage = inlier_count / inliers.numel()
                # Ensure all metrics are CPU scalars (not GPU tensors)
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
                        "HyperParameters/LR": self.optimizer.param_groups[0]["lr"],
                        f"Step/{'val' if VALIDATION else ''}batch": self.step[
                            f"{PHASE}_batch"
                        ],
                        f"Step/{'idx' if TEST else 'epoch'}": epoch,
                    }
                )

                # Store images separately to avoid keeping them in DataFrame
                images_for_wandb = None
                if log_images_this_batch:
                    # Adding logged images only if is time to do so
                    # Move tensors to CPU before visualization to prevent GPU memory accumulation
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

                # Updating the local metrics dataframe (without images to save memory)
                # Use a more memory-efficient approach: create new row dict and append
                metrics_row = {k: v for k, v in metrics.items()}
                new_row_df = pd.DataFrame([metrics_row])
                self.metrics[PHASE] = pd.concat(
                    [self.metrics[PHASE], new_row_df],
                    ignore_index=True,
                )
                # Explicitly delete the intermediate DataFrame to free memory
                del new_row_df, metrics_row

                ### LOGGING
                self.console_log_metrics(
                    stage=PHASE,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    dataloader_len=len(dataloader),
                    extra_info="W" if step < self.warmup_steps else None,
                )
                if tracking_result is not None:
                    tm = tracking_result["metrics"]
                    self.logger.info(
                        f"[Tracking] "
                        f"cycle={tm['cycle_error']:.3f} "
                        f"desc={tm['loss_desc']:.4f} "
                        f"c2f={tm['coarse_to_fine_delta']:.2f} "
                        f"conf={tm['confidence_mean']:.3f} "
                        f"vis={tm.get('visibility_ratio', -1):.3f} "
                        f"loss={tm['loss_total']:.4f}",
                        context=PHASE.upper(),
                    )
                if self.wandb is not None:
                    # Ensure all wandb metrics are CPU-based (wandb handles PIL Images correctly)
                    wandb_metrics = metrics_for_wandb(metrics, PHASE)
                    wandb_metrics = self._namespace_refinement_metrics(
                        wandb_metrics, PHASE
                    )
                    # Add images separately if they exist
                    if images_for_wandb is not None:
                        wandb_images = metrics_for_wandb(images_for_wandb, f"{PHASE}/images")
                        wandb_metrics.update(wandb_images)
                        del wandb_images  # Explicitly delete to free memory
                    self.wandb.log(wandb_metrics)
                    del wandb_metrics  # Explicitly delete to free memory
                # Clean up images even if wandb is None
                if images_for_wandb is not None:
                    del images_for_wandb
                self.log_loaded_paths(paths, PHASE)

                ### CLEANUP
                # Explicitly delete all tensors to free GPU memory
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
                # Clear loss-related variables
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

        ### LOGGING Epoch metrics
        epochstr = (
            "idx" if TEST else "epoch"
        )  # Tests need to have the test index in the key
        # Compute epoch metrics and ensure they're scalars (not keeping DataFrame references)
        epoch_df_slice = self.metrics[PHASE][
            self.metrics[PHASE][f"Step/{epochstr}"] == epoch
        ]
        epoch_mean_series = epoch_df_slice.mean()
        # Convert to dict immediately to release Series reference
        epoch_mean_dict = epoch_mean_series.to_dict()
        del epoch_df_slice, epoch_mean_series  # Explicitly delete to free memory

        epoch_metrics = metrics_for_wandb(epoch_mean_dict, PHASE)
        del epoch_mean_dict  # Explicitly delete to free memory

        # Assuming epoch_metrics is already defined
        epoch_metrics = {
            key.replace(PHASE, f"{PHASE}/{epochstr}"): value
            for key, value in epoch_metrics.items()
            if PHASE in key
        }
        epoch_metrics = self._namespace_refinement_metrics(
            epoch_metrics, f"{PHASE}/{epochstr}"
        )

        # Extract loss value for early stopping before deleting epoch_metrics
        validation_loss = None
        if VALIDATION:
            loss_key = f"{PHASE}/epoch/Loss"
            if loss_key in epoch_metrics:
                validation_loss = float(epoch_metrics[loss_key])

        if self.wandb is not None:
            self.wandb.log(epoch_metrics)
            del epoch_metrics  # Explicitly delete after logging

        torch.cuda.empty_cache()
        gc.collect()

        # Periodic cleanup: limit metrics DataFrame size to prevent unbounded growth
        # Keep only last 1000 batches per phase to prevent memory accumulation
        max_metrics_rows = 1000
        if len(self.metrics[PHASE]) > max_metrics_rows:
            # Keep only the most recent rows
            self.metrics[PHASE] = (
                self.metrics[PHASE].tail(max_metrics_rows).reset_index(drop=True)
            )
            gc.collect()  # Force garbage collection after DataFrame truncation
        ### EARLY-STOPPING
        if VALIDATION:
            if validation_loss is None:
                # Fallback: try to get loss from metrics DataFrame if epoch_metrics was deleted
                epoch_df_slice = self.metrics[PHASE][
                    self.metrics[PHASE][f"Step/{epochstr}"] == epoch
                ]
                if "Loss" in epoch_df_slice.columns and len(epoch_df_slice) > 0:
                    validation_loss = float(epoch_df_slice["Loss"].mean())
                else:
                    raise ValueError(f"Cannot find validation loss for epoch {epoch}")
            self.step[epochstr] += 1  # Increasing epoch/idx counter
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

        This context manager ensures that gradients are only computed during training
        and not during validation or testing phases.

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

        This method handles the complete backward pass including gradient computation,
        gradient clipping for stability, and optimizer stepping. It also includes
        comprehensive error handling and memory management.

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

        # Step only if warmup phase is finished and we are backpropagating the accumulated gradients
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

        This method implements a two-stage optimization strategy where different
        optimizers are used for different phases of training (bootstrap vs refining).

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
                self.matcher.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
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
        from dataset.stereomis_tracking import StereoMISTracking
        from matching.tracking_metrics import compute_tracking_metrics
        from utilities.tracking_evaluation import (
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

            # Pre-load all frames if using TemporalTracker (reused for GT eval below)
            _cached_all_frames = None
            if has_temporal_tracker:
                _cached_all_frames = torch.stack(
                    [ds[t]["image"] for t in range(len(ds))], dim=0,
                ).unsqueeze(0).to(self.device)  # [1, T, 3, H, W]

            # --- Dense grid tracking (for qualitative video) ---
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

            # --- Render qualitative video ---
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

            # --- Log video to wandb ---
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

            # --- GT-based metrics + GT tracks video ---
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

                results = compute_tracking_metrics(
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

                # Per-point per-frame L2 error: [N_pts, T_pred]
                pixel_errors = (pred_tracks - gt_tracks).norm(dim=-1)  # [N_pts, T_pred]

                # --- Render comparison video (GT white + predicted error-colored) ---
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

        # --- Aggregate metrics across sequences ---
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

            # Save tracking metrics CSV
            tracking_csv_path = os.path.join(tracking_dir, "tracking_metrics.csv")
            pd.DataFrame(seqs_with_metrics).to_csv(tracking_csv_path, index=False)
            self.logger.info(
                f"  Tracking metrics CSV: {os.path.abspath(tracking_csv_path)}",
                context="TEST",
            )

    def log_tests(self) -> None:
        """
        Logs the test metrics to Weights and Biases.

        This method provides comprehensive test result logging including:
        - Statistical summary of test metrics
        - CSV export of detailed results
        - WandB table logging for visualization
        - File location logging for easy access
        """

        self.logger.info(">> TEST REPORT", context="TEST")
        test_describe = self.metrics["Test"].describe().to_string()
        self.logger.info("\n" + test_describe, context="TEST")
        del test_describe  # Explicitly delete to free memory
        self.metrics["Test"].to_csv(self.TEST_DIR + "/test_metrics.csv")
        # Create wandb table and immediately delete reference
        test_table = wandb.Table(dataframe=self.metrics["Test"])
        self.wandb.log({"Test/Summary": test_table})
        del test_table  # Explicitly delete to free memory

        # Log locations of important data
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

        # Log WandB URLs again for convenience
        if hasattr(self.wandb, "url") and self.wandb.url:
            self.logger.info(f"WandB run URL: {self.wandb.url}", context="WANDB")
            project_url = self.wandb.url.rsplit("/", 1)[0]
            self.logger.info(f"WandB project URL: {project_url}", context="WANDB")

        # Remove unused IMAGES_DIR if it exists and is empty
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

        This method exports the accumulated metrics dataframes to CSV files
        for persistent storage and external analysis.
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
        _, shared_pca = embedding2color(shared_embeddings, pca=None)
        source_rgb = embedding2color(
            source_embedding, pca=shared_pca
        )  # [1, 3, H_p, W_p]
        target_rgb = embedding2color(
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

        This method provides real-time console output of training progress including
        current metrics, epoch/batch progress, and any additional status information.

        Args:
            stage: The current stage ('Training', 'Validation', or 'Test').
            epoch: Current epoch number.
            batch_idx: Current batch index.
            dataloader_len: Length of the dataloader being used.
            extra_info: Additional information to display in the phase indicator.
        """
        # Define the phase indicator based on stage
        # if stage == "Training":
        #     epoch_batch_info = align(
        #         f"E {str(epoch+1)}/{self.epochs}", 8, "right"
        #     ) + align(f"B{str(batch_idx+1)}/{dataloader_len}", 8, "right")
        # elif stage == "Validation":
        #     epoch_batch_info = align(
        #         f"E {str(epoch+1)}/{self.epochs}", 8, "right"
        #     ) + align(f"B{str(batch_idx+1)}/{dataloader_len}", 8, "right")
        # elif "Test" in stage:
        #     test_idx = epoch
        # epoch_batch_info = f"E {int(test_idx)} " + align(
        #     f"B{str(batch_idx+1)}/{dataloader_len}", 8, "right"
        # )
        epoch_batch_info = align(
            f"E {str(epoch + 1)}/{self.epochs} ", 10, "right"
        ) + align(f"B {str(batch_idx + 1)}/{dataloader_len} ", 10, "left")
        if extra_info is not None:
            phase_indicator = f"[purple ]{extra_info}[/purple ]"  # + phase_indicator
        # Print header with run name and status information
        if "offline" in self.runname:
            printedrunname = "run"
        else:
            printedrunname = f"{self.runname.split('-')[0][0]}{self.runname.split('-')[1][0]}{self.runname.split('-')[2]}"
        metricstring = (
            align(f"{printedrunname}:", 6, "right")
            # + f"|"
            # + align(
            #     f"{phase_indicator}{self.step[f'{stage}_batch'] if stage != 'Test' else ''}",
            #     24,  # if "Test" in stage else 21,
            #     "center",
            # )
            # + f"|"
            + epoch_batch_info
        )

        # Generate metrics string
        metrs = ""
        # Print metrics from the appropriate metrics dictionary
        if stage in self.metrics.keys():
            # First print the Loss column if it exists
            if (
                "Loss" in self.metrics[stage].columns
                and self.metrics[stage]["Loss"].iloc[-1] is not None
            ):
                metrs += (
                    f"[yellow]Loss[/yellow]"
                    + "="
                    + align(
                        f"{self.metrics[stage]['Loss'].iloc[-1]:.4f}",
                        6,
                        "left",
                    )
                    + " "
                )

            # Then print other columns that don't have a "/" in their name
            for m in self.metrics[stage].columns:
                if (
                    m != "Loss"
                    and "/" not in m
                    and self.metrics[stage][m].iloc[-1] is not None
                ):
                    metrs += (
                        f"[yellow]{m[:4]}[/yellow]"
                        + "="
                        + align(
                            f"{self.metrics[stage][m].iloc[-1]:.4f}",
                            6,
                            "left",
                        )
                        + " "
                    )
        self.logger.info(metricstring + metrs, context=stage.upper())

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

        This function generates visualization images for patch matches, pixel matches,
        epipolar geometry, and mined triplets from the current batch of data.

        Args:
            framestack: Tensor containing the source frames.
            warped: Tensor containing the warped (target) frames.
            sim_matrix: Similarity matrix between patches.
            source_pixels_matched: Coordinates of matched pixels in the source image.
            target_pixels_matched: Coordinates of matched pixels in the target image.
            true_pixels_matched: Ground truth coordinates of matches in the target image.
            scores: Confidence scores for the matches.
            batch_idx_match: Batch indices for each match.
            triplets_dict: Dictionary containing triplet information (anchor, positive, negative indices).
            fundamental_pred: Predicted fundamental matrices.
            patch_size: Size of the patches used for matching.
            topk: Number of top matches to visualize.
            batch_idx: Batch index to visualize, defaults to 0.

        Returns:
            Dictionary of PIL image objects for visualization.
        """
        # Filter data for the specified batch index
        if patch_size is None:
            patch_size = self.patch_size
        batch_filter = batch_idx_match == batch_idx
        if triplets_dict is not None:
            batch_triplet_filter = triplets_dict["batch_indices"] == batch_idx
        TEST = triplets_dict is None
        # Create visualization dictionary
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

    def _tracking_training_step(self, tracking_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Run a single tracking training step over a temporal window.

        Args:
            tracking_batch: Dict with keys ``frames`` [B, T, 3, H, W],
                ``query_pts`` [B, Q, 2], ``gt_tracks`` [B, Q, T, 2],
                ``gt_vis`` [B, Q, T].

        Returns:
            Scalar tracking loss (position + visibility).
        """
        frames = tracking_batch["frames"].to(self.device)      # [B, T, 3, H, W]
        query_pts = tracking_batch["query_pts"].to(self.device) # [B, Q, 2]
        gt_tracks = tracking_batch["gt_tracks"].to(self.device) # [B, Q, T, 2]
        gt_vis = tracking_batch["gt_vis"].to(self.device)       # [B, Q, T]

        B, T_w, _, H, W = frames.shape
        Q = query_pts.shape[1]

        current_pts = query_pts.clone()  # [B, Q, 2]
        all_pred_pos = [current_pts.unsqueeze(2)]  # list of [B, Q, 1, 2]
        all_vis_logits = []

        for t in range(1, T_w):
            track_out = self.matcher.track_points(
                current_pts, frames[:, t - 1], frames[:, t]
            )
            current_pts = track_out["tracked_points"]  # [B, Q, 2]
            all_pred_pos.append(current_pts.unsqueeze(2))

            vis_logit = track_out.get("visibility_logit")
            if vis_logit is not None:
                all_vis_logits.append(vis_logit.squeeze(-1).unsqueeze(2))  # [B, Q, 1]

        pred_tracks = torch.cat(all_pred_pos, dim=2)  # [B, Q, T_w, 2]

        pos_weight = float(self.config.get("TRACKING_POSITION_LOSS_WEIGHT", 1.0))
        vis_weight = float(self.config.get("TRACKING_VISIBILITY_LOSS_WEIGHT", 0.5))

        pos_loss = self.tracking_pos_loss(pred_tracks, gt_tracks, gt_vis) * pos_weight

        vis_loss = torch.tensor(0.0, device=self.device)
        if len(all_vis_logits) > 0:
            pred_vis_logits = torch.cat(all_vis_logits, dim=2)  # [B, Q, T_w-1]
            vis_loss = self.tracking_vis_loss(pred_vis_logits, gt_vis[:, :, 1:]) * vis_weight

        return pos_loss + vis_loss

    # ==================================================================
    # Phase 2: Temporal Tracking Training
    # ==================================================================

    def _init_tracking_phase(self) -> None:
        """Initialize Phase 2 components: TemporalTracker + sequence dataloader.

        Called once at the start of ``tracking_trainloop``.
        """
        # Build the TemporalTracker around the existing matcher
        self.temporal_tracker = TemporalTracker.from_config(
            self.matcher, self.config,
        )

        # Load pretrained descriptor checkpoint if specified
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

        # Freeze matcher, only train refinement network
        self.temporal_tracker._freeze_matcher_params()
        self.temporal_tracker.refinement_net.to(self.device)

        # Create optimizer for refinement net only
        self.tracking_optimizer = torch.optim.Adam(
            self.temporal_tracker.refinement_net.parameters(),
            lr=self.config.get("LEARNING_RATE", 1e-4),
            weight_decay=self.config.get("WEIGHT_DECAY", 0),
        )

        # Build sequence datasets from existing training data
        from dataset.sequence_sampler import SequenceWindowDataset

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
                pass  # Sequence dataset doesn't use samplers

            self.csv_log_metrics()
            torch.cuda.empty_cache()
            gc.collect()

            if val_status == "EARLYSTOP":
                break

        # Save final tracking checkpoint
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
            return "EMPTY"

        num_query_pts = int(self.config.get("TRACKING_NUM_QUERY_POINTS", 64))
        epoch_losses = []

        ctx = nullcontext() if TRAINING else torch.no_grad()
        with ctx:
            for batch_idx, batch in enumerate(dataloader):
                frames = batch["frames"].to(self.device)  # [B, T, 3, H, W]

                loss_dict = self.temporal_tracker.training_step(
                    frames, self.config, num_query_points=num_query_pts,
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
                metrics["VisReg"] = m.get("loss_vis_reg", 0.0)

                # Store metrics
                new_row = pd.DataFrame([metrics])
                self.metrics[phase] = pd.concat(
                    [self.metrics[phase], new_row], ignore_index=True,
                )
                epoch_losses.append(loss_total.item())

                # Console logging (every batch, same style as pretrain)
                self.console_log_metrics(
                    stage=phase,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    dataloader_len=len(dataloader),
                )

                # W&B logging
                if self.wandb is not None and batch_idx % self.logfreq_wandb == 0:
                    wandb_metrics = {k: v for k, v in metrics.items() if v is not None}
                    self.wandb.log(wandb_metrics)

                del frames, loss_dict, loss_total
                torch.cuda.empty_cache()

        # Epoch-level metrics
        if epoch_losses:
            mean_loss = np.mean(epoch_losses)
            self.logger.info(
                f">> {phase} epoch {epoch+1} mean loss: {mean_loss:.4f}",
                context="TRACKING",
            )
            if self.wandb is not None:
                self.wandb.log({f"{phase}/tracking/epoch_loss": mean_loss})

            # Early stopping (validation only)
            if not TRAINING:
                self.step["epoch"] += 1
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

        This method loads the model weights from the most recent checkpoint
        and logs the successful restoration.
        """
        # Reinstantiate the model from the latest checkpoint
        self.logger.info(
            f"Attempting reinstatiation from checkpoint @ {self.MODELS_DIR}",
            context="GCLOUD",
        )
        self.matcher.model.fromArtifact(model_name=self.runname, local_path=self.MODELS_DIR)
        self.logger.info(
            f"Model reinstantiated from checkpoint @ {self.runname}",
            context="GCLOUD",
        )
