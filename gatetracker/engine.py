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
import math
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
from gatetracker.utils.schedules import _linear_ramp, _piecewise_linear_epochs
from gatetracker.distributed_context import (
    all_reduce_sum_scalars,
    barrier,
    ddp_find_unused_parameters,
    ddp_static_graph,
    is_ddp_enabled,
    is_main_process,
    ShardedListSampler,
    unwrap_model,
)
from gatetracker.env_bootstrap import (
    pretrained_checkpoint_path_candidates,
    resolve_dataset_filesystem_path,
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

        device_dirs = initialize.device_and_directories(config)
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
        self._wrap_matcher_parallel()
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
        init(initialize.setup_run_directories, self.RUNS_DIR, self.wandb, False, config)
        self.config["name"] = self.runname

        geometry_model_name = self.config.get("GEOMETRY_MODEL_NAME", "Ruicheng/moge-2-vits-normal")
        self.geometryPipeline = GeometryPipeline(
            geometry_model_name=geometry_model_name,
            device=str(self.device),
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
        # Monotonic W&B x-axis for ``run_tests`` / eval media (``step/test_eval``).
        self._wandb_test_eval_step = 0
        self.current_epoch = 0
        self.global_step = 0

        # Hierarchical-fusion gate freeze state (see ``_apply_gate_freeze``).
        # Gate params keep ``requires_grad=True`` so the DDP autograd graph is
        # stable under ``static_graph=True``; during the freeze window we simply
        # drop their gradients before ``optimizer.step()`` so no weight updates
        # occur. Matches the pre-fix semantics while preserving static-graph.
        self._gate_freeze_active: bool = False
        self._gate_freeze_params: List[torch.nn.Parameter] = []

        self.refinement_metric_names = {
            "RefinementActiveFraction",
            "RefinementOffsetMean",
            "RefinementScoreMean",
            "CoarseErrorMean",
            "RefinedErrorMean",
            "RefinementGainPx",
            "RefinementWinRate",
        }

    def _wrap_matcher_parallel(self) -> None:
        """Wrap ``self.matcher.model`` in DDP or DataParallel when configured."""
        import torch.nn as nn

        from gatetracker.distributed_context import is_ddp_enabled, is_dp_enabled

        if is_ddp_enabled(self.config):
            lr = int(self.config.get("LOCAL_RANK", 0))
            self.matcher.model = nn.parallel.DistributedDataParallel(
                self.matcher.model,
                device_ids=[lr],
                output_device=lr,
                find_unused_parameters=ddp_find_unused_parameters(self.config),
                static_graph=ddp_static_graph(self.config),
            )
        elif is_dp_enabled(self.config):
            if self.device.type == "cuda" and torch.cuda.device_count() > 1:
                self.matcher.model = nn.DataParallel(self.matcher.model)

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

        for candidate in pretrained_checkpoint_path_candidates(ref, runs_dir=self.RUNS_DIR):
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
            unwrap_model(self.matcher.model).fromArtifact(pth_namestring=resolved_path)
            self.logger.info(
                f"Model loaded from checkpoint: {resolved_path}", context="LOAD"
            )
            return

        unwrap_model(self.matcher.model).fromArtifact(model_name=checkpoint_ref)
        self.logger.info(
            f"Model loaded from artifact/model name: {checkpoint_ref}", context="LOAD"
        )

    def _wandb_epoch_axis_dict(self, epoch: int) -> Dict[str, int]:
        """W&B x-axis payload for epoch-level scalars (``define_metric`` → ``step/epoch``)."""
        return {"step/epoch": int(epoch)}

    def _wandb_test_eval_step_dict(self) -> Dict[str, int]:
        """W&B x-axis for post-hoc test tracking / summary logs (``step/test_eval``)."""
        self._wandb_test_eval_step += 1
        return {"step/test_eval": self._wandb_test_eval_step}

    def _track_long_sequence_infer_kwargs(self) -> Dict[str, Any]:
        """Keyword args for ``TemporalTracker.track_long_sequence`` from config."""
        kwargs: Dict[str, Any] = {
            "vis_agg": str(self.config.get("TRACKING_LONGSEQ_VIS_AGG", "mean")),
            "infer_max_step_px": float(self.config.get("TRACKING_INFER_MAX_STEP_PX", 0.0)),
            "centrality_weighting": bool(
                self.config.get("TRACKING_LONGSEQ_CENTRALITY", False),
            ),
            "anchor_bank_size": int(self.config.get("TRACKING_ANCHOR_BANK_SIZE", 1)),
            "anchor_refresh_vis_thresh": float(
                self.config.get("TRACKING_ANCHOR_REFRESH_VIS_THRESH", 0.9),
            ),
            "redetection": bool(self.config.get("TRACKING_REDETECTION", False)),
            "redetect_after": int(self.config.get("TRACKING_REDETECT_AFTER", 5)),
            "redetect_topk": int(self.config.get("TRACKING_REDETECT_TOPK", 8)),
            "redetect_vis_thresh": float(
                self.config.get("TRACKING_REDETECT_VIS_THRESH", 0.3),
            ),
        }
        iters_infer = self.config.get("TEMPORAL_REFINEMENT_NUM_ITERS_INFER", None)
        if iters_infer is not None:
            kwargs["num_iters"] = int(iters_infer)
        return kwargs

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

    def _optimizer_lr_metrics(self, phase: str) -> Dict[str, float]:
        """Scalar LR keys for logging (primary group + per-group names when present).

        Keys are phase-prefixed so W&B can bind LR to ``step/val_batch`` vs
        ``step/train_batch`` via ``{phase}/optim/*`` (see ``register_wandb_step_axes``).
        """
        out: Dict[str, float] = {
            f"{phase}/optim/LR": float(self.optimizer.param_groups[0]["lr"]),
        }
        for i, g in enumerate(self.optimizer.param_groups):
            gname = g.get("group_name", f"group_{i}")
            out[f"{phase}/optim/LR/{gname}"] = float(g["lr"])
        return out

    def _anneal_gate_temperature(self, epoch: int) -> None:
        """Linearly anneal the hierarchical-fusion gate temperature.

        Interpolates ``FUSION_GATE_TEMPERATURE`` (T0, warm/smooth) to
        ``FUSION_GATE_TEMPERATURE_FINAL`` (T_final, sharp) over the first
        ``FUSION_GATE_ANNEAL_EPOCHS`` training epochs, then holds at T_final.
        When T0 == T_final (or the module is absent) this is a no-op, so the
        behavior remains backward compatible with configs that do not set
        these keys.
        """
        model = unwrap_model(self.matcher.model)
        fusion = getattr(model, "hierarchical_fusion", None)
        if fusion is None or not hasattr(fusion, "set_gate_temperature"):
            return

        T0 = float(self.config.get("FUSION_GATE_TEMPERATURE", fusion.gate_temperature))
        T_final = float(self.config.get("FUSION_GATE_TEMPERATURE_FINAL", T0))
        if T_final == T0:
            fusion.set_gate_temperature(T0)
            return

        anneal_epochs = int(self.config.get("FUSION_GATE_ANNEAL_EPOCHS", 10))
        frac = min(1.0, max(0, epoch) / max(anneal_epochs, 1))
        T = T0 + (T_final - T0) * frac
        fusion.set_gate_temperature(T)

    def _apply_gate_freeze(self, epoch: int) -> None:
        """Freeze the hierarchical-fusion gate parameters for the first
        ``GATE_FREEZE_EPOCHS`` training epochs.

        During the freeze window, ``local_gates`` and ``register_gates``
        behave as if frozen (no weight updates), forcing descriptors and
        refinement to work across all DINOv3 layers with a near-uniform gate
        (after the register-gate zero-init + tanh clamp in ``fusion.py``).
        Once the epoch counter crosses the threshold, gate parameters start
        updating under the entropy regularizer.

        Implementation: we keep ``requires_grad = True`` on the gate
        parameters at all times, and instead drop their gradients
        (``p.grad = None``) in ``backward_pass`` while the freeze window is
        active. This is functionally equivalent to ``requires_grad = False``
        in terms of weight updates (all PyTorch optimizers skip params whose
        ``.grad`` is ``None``), but keeps the set of grad-producing
        parameters constant across iterations, which is required by
        ``DistributedDataParallel(static_graph=True)``. Toggling
        ``requires_grad`` at the freeze boundary instead would change the
        parameters that participate in the DDP reducer and trigger
        ``RuntimeError: Your training graph has changed in this iteration``
        on the first batch after the transition.

        A ``GATE_FREEZE_EPOCHS`` value of ``0`` (default when the key is
        absent from the config) is a no-op for every epoch, preserving
        backward compatibility with existing checkpoints / configs.
        """
        gate_freeze_epochs = int(self.config.get("GATE_FREEZE_EPOCHS", 0))
        if gate_freeze_epochs <= 0:
            self._gate_freeze_active = False
            self._gate_freeze_params = []
            return
        model = unwrap_model(self.matcher.model)
        fusion = getattr(model, "hierarchical_fusion", None)
        if fusion is None:
            self._gate_freeze_active = False
            self._gate_freeze_params = []
            return

        gate_params: List[torch.nn.Parameter] = []
        for gate_list_name in ("local_gates", "register_gates"):
            gate_list = getattr(fusion, gate_list_name, None)
            if gate_list is None:
                continue
            for param in gate_list.parameters():
                # Keep requires_grad=True for a stable DDP static graph; the
                # freeze is enforced via grad-masking in ``backward_pass``.
                param.requires_grad_(True)
                gate_params.append(param)

        self._gate_freeze_active = epoch < gate_freeze_epochs
        self._gate_freeze_params = gate_params

    def _compute_gate_entropy_loss(
        self,
    ) -> Tuple[Optional[torch.Tensor], Optional[float]]:
        """
        Entropy-minimization regularizer on the hierarchical-fusion gate weights.

        For normalized softmax weights :math:`w_\\ell \\in \\Delta^{L-1}` at each
        patch location, we compute the mean of

        .. math::
            \\tilde H = \\frac{-\\sum_\\ell w_\\ell \\log w_\\ell}{\\log L}\\in[0,1],

        averaged over source + target images, batch and patch locations. The
        returned tensor is :math:`\\lambda_{gate} \\cdot \\tilde H` using
        ``GATE_ENTROPY_WEIGHT`` from config (default 0.0 -> disabled).

        Returns ``(loss_tensor, loss_value)``. Both are ``None`` when the
        weight is 0, no diagnostics are available, or gate weights are
        detached (e.g. under ``torch.no_grad``).
        """
        weight = float(self.config.get("GATE_ENTROPY_WEIGHT", 0.0))
        if weight <= 0.0:
            return None, None

        model = unwrap_model(self.matcher.model)
        diagnostics = getattr(model, "latest_diagnostics", None) or {}

        tensors = []
        for side in ("source", "target"):
            diag = diagnostics.get(side)
            if diag is None:
                continue
            lw = diag.get("layer_weights")  # [B, L, N]
            if lw is None or not torch.is_tensor(lw) or lw.grad_fn is None:
                continue
            tensors.append(lw)
        if not tensors:
            return None, None

        layer_weights = torch.cat(tensors, dim=0).clamp_min(1e-8)  # [B_all, L, N]
        num_layers = layer_weights.shape[1]
        log_L = float(np.log(max(num_layers, 2)))
        entropy = -(layer_weights * layer_weights.log()).sum(dim=1) / log_L  # [B_all, N]
        loss = entropy.mean()
        return weight * loss, loss.detach().item()

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

            if is_ddp_enabled(self.config):
                nxt = int(e) + 1
                self.dataset["Training"].reset_sampler(next_shuffle_epoch=nxt)
                self.dataset["Validation"].reset_sampler(next_shuffle_epoch=nxt)
            else:
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
                unwrap_model(self.temporal_tracker.refinement_net).load_state_dict(
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
        self.current_epoch = int(epoch)
        if TRAINING:
            self.matcher.model.train()
        else:
            self.matcher.model.eval()
        images_logged = False
        dataset = self.dataset[PHASE]

        if is_ddp_enabled(self.config):
            r = int(self.config.get("RANK", 0))
            ws = int(self.config.get("WORLD_SIZE", 1))
            _sampler = ShardedListSampler(dataset, r, ws)
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.config.WORKERS,
                drop_last=True,
                sampler=_sampler,
                pin_memory=self.config.PIN_MEMORY,
                prefetch_factor=self.config.PREFETCH_FACTOR,
                collate_fn=collate_fn,
            )
        else:
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

        self._anneal_gate_temperature(epoch)
        if TRAINING:
            self._apply_gate_freeze(epoch)

        if len(dataloader) == 0:
            if is_main_process():
                print("[DATASET]: Empty dataloader, skipping epoch.")
            return "EMPTY"

        with self.choose_if_grad(PHASE):
            for batch_idx, sample in enumerate(dataloader):
                step = epoch * len(dataloader) + batch_idx
                self.global_step = int(step)
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
                # Target-only photometric noise: adds brightness / contrast jitter,
                # per-pixel Gaussian noise, mild blur and gamma shifts to the
                # NVS-warped target view (``synthetic_framestack[:, 1]``) so the
                # descriptor head sees appearance variation that mimics real
                # frame-to-frame differences. Disabled at validation / test time.
                if TRAINING:
                    synthetic_framestack = photometric_noise_augmentation(
                        synthetic_framestack,
                        target_only=True,
                    )
                descriptors = self.matcher.model(synthetic_framestack)

                loss_value = None
                descriptor_loss_value = None
                refinement_loss_value = None
                refinement_active_matches = None
                refinement_weight_mean = None
                gate_entropy_value = None
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

                    gate_entropy_tensor, gate_entropy_value = (
                        self._compute_gate_entropy_loss()
                    )
                    if gate_entropy_tensor is not None:
                        loss_tensor = loss_tensor + (
                            gate_entropy_tensor / self.gradient_accumulation_steps
                        )

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
                    if pairwise_tracking_enabled(self.config):
                        tracking_result = compute_pairwise_tracking_losses(
                            matching_pipeline=self.matcher,
                            source_image=synthetic_framestack[:, 0],
                            target_image=synthetic_framestack[:, -1],
                            model_output=descriptors,
                            config=self.config,
                        )
                        tracking_loss_value = tracking_result["loss_total"].detach().item()
                torch.cuda.empty_cache()

                metrics = self.matcher.compute_metrics(
                    source_pixels_matched,
                    target_pixels_matched,
                    true_pixels_matched,
                    batch_idx_match,
                    scores,
                    fundamental_pred,
                    fundamental_gt,
                    descriptor_scores=descriptor_scores,
                )
                inlier_percentage = inliers.count_nonzero().item() / inliers.numel()
                metrics = {
                    k: (v.item() if isinstance(v, torch.Tensor) else v)
                    for k, v in metrics.items()
                }
                architecture_metrics = self.compute_architecture_metrics(
                    descriptors=descriptors,
                    triplets_dict=triplets,
                )
                metrics.update({f"{PHASE}/{k}": v for k, v in architecture_metrics.items()})

                metrics.update(
                    {
                        # Unified Loss = EpipolarError (pixels) in both phases so that
                        # `Training/epoch/Loss` and `Validation/epoch/Loss` are directly
                        # comparable. The optimization target `loss_tensor` (descriptor +
                        # refinement + optional tracking/gate-entropy) is unchanged; its
                        # scalar is logged separately as `TotalLoss` for training.
                        "Loss": metrics.get("EpipolarError"),
                        "PrimaryMetric": metrics.get("EpipolarError"),
                        "TotalLoss": loss_value if TRAINING else None,
                        "DescriptorLoss": descriptor_loss_value if TRAINING else None,
                        "TrackingLoss": tracking_loss_value if TRAINING else None,
                        "RefinementLoss": refinement_loss_value if TRAINING else None,
                        "GateEntropyLoss": gate_entropy_value if TRAINING else None,
                        "NRefinementSupervised": (
                            refinement_active_matches if TRAINING else None
                        ),
                        "RefinementWeightMean": (
                            refinement_weight_mean if TRAINING else None
                        ),
                        "InlierPercentage": inlier_percentage,
                        "NTripletsMined": (
                            (descriptor_pairs_mined / self.batch_size)
                            if TRAINING
                            else None
                        ),
                        f"{PHASE}/optim/GradNorm": backward_output.get("grad_norm"),
                        f"{PHASE}/optim/GradNormClipped": backward_output.get(
                            "grad_norm_clipped"
                        ),
                        f"{PHASE}/optim/WeightNorm": backward_output.get("weight_norm"),
                        **{
                            f"{PHASE}/optim/GradNorm/{gname}": gval
                            for gname, gval in (
                                backward_output.get("per_group_grad_norms") or {}
                            ).items()
                        },
                        **self._optimizer_lr_metrics(PHASE),
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
                if tracking_result is not None:
                    for _tk, _tv in tracking_result["metrics"].items():
                        metrics_row[f"{PHASE}/tracking/{_tk}"] = _tv
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
                    tracking_metrics=(
                        tracking_result["metrics"] if tracking_result is not None else None
                    ),
                )

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
        # Drop columns whose values are entirely None/NaN for this phase+epoch
        # slice (e.g. `DescriptorLoss`, `RefinementLoss`, `TrackingLoss`,
        # `GateEntropyLoss`, `MDistMean`, ... during Validation). Without this
        # filter pandas propagates them as "NaN" strings into W&B.
        epoch_df_slice = epoch_df_slice.dropna(axis=1, how="all")
        epoch_mean_series = epoch_df_slice.mean(numeric_only=True)
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
        loss_key = f"{PHASE}/epoch/Loss"
        if VALIDATION:
            epoch_df_val = self.metrics[PHASE][
                self.metrics[PHASE][f"Step/{epochstr}"] == epoch
            ]
            if "Loss" in epoch_df_val.columns and len(epoch_df_val) > 0:
                loss_sum = float(epoch_df_val["Loss"].sum())
                cnt = float(len(epoch_df_val))
            else:
                loss_sum, cnt = 0.0, 0.0
            if is_ddp_enabled(self.config):
                loss_sum, cnt = all_reduce_sum_scalars(loss_sum, cnt, self.device)
            if cnt <= 0:
                raise ValueError(f"Cannot find validation loss for epoch {epoch}")
            validation_loss = loss_sum / cnt
            if loss_key in epoch_metrics:
                epoch_metrics[loss_key] = validation_loss

        if self.wandb is not None:
            epoch_metrics.update(self._wandb_epoch_axis_dict(epoch))
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
            self.step[epochstr] += 1
            if self.LRschedulerPlateau is not None:
                self.LRschedulerPlateau.step(validation_loss)
            should_stop = self.earlystopping(
                validation_loss,
                self.matcher.model,
                epoch,
            )
            if should_stop:
                if is_main_process():
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
                "grad_norm_clipped": np.nan,
                "weight_norm": np.nan,
                "per_group_grad_norms": {},
            }

        ERROR_IN_BACKWARD_PASS = False
        try:
            loss_tensor.backward()
        except RuntimeError as e:
            print(
                f">> [ERROR]: {e} - Skipping batch {self.step['Training_batch']} in epoch {self.step['epoch']}"
            )
            ERROR_IN_BACKWARD_PASS = True

        # Hierarchical-fusion gate freeze: drop grads on gate params during the
        # freeze window so the optimizer (which skips params with ``grad is
        # None``) leaves their weights untouched. The params stay
        # ``requires_grad=True`` so DDP's static graph remains valid.
        if self._gate_freeze_active and self._gate_freeze_params:
            for p in self._gate_freeze_params:
                p.grad = None

        grad_norm, weight_norm = optimization.get_grad_and_weight_norms(
            self.matcher.model.parameters()
        )
        per_group_grad_norms = {
            g.get("group_name", f"group_{i}"): optimization.get_grad_and_weight_norms(
                g["params"]
            )[0]
            for i, g in enumerate(self.optimizer.param_groups)
        }
        grad_clip_max = float(self.config.get("GRAD_CLIP_NORM", 1.0))
        clipped_norm = torch.nn.utils.clip_grad_norm_(
            self.matcher.model.parameters(), max_norm=grad_clip_max
        )
        clipped_norm_val = (
            clipped_norm.item() if torch.is_tensor(clipped_norm) else float(clipped_norm)
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
            "grad_norm_clipped": clipped_norm_val,
            "weight_norm": weight_norm,
            "per_group_grad_norms": per_group_grad_norms,
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
            if is_main_process():
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
        Run point tracking evaluation on StereoMIS and/or STIR test sequences.

        Respects ``DATASETS.<NAME>.TEST_VIDEOS`` (merged with optional ``TRACKING``
        phase block via ``merge_phase_video_list``). When ``TEST_VIDEOS`` is empty
        or unset, all discovered sequences are used (backward compatible).

        Logs under ``Test/tracking/eval/stereomis_annotated/...`` and
        ``Test/tracking/eval/stir/...``.
        """
        import random
        from dataset.loader_phase import merge_phase_video_list
        from dataset.utils import select_videos_by_patterns
        from gatetracker.data import STIRTracking, StereoMISTracking
        from gatetracker.tracking.metrics import (
            STIR_THRESHOLDS_PX,
            compute_stir_endpoint_metrics,
            compute_tap_metrics,
        )
        from gatetracker.utils.visualization import (
            infer_tracks,
            infer_tracks_windowed,
            make_grid_points,
            render_comparison_video,
            render_tracks_video,
        )

        datasets_config = self.config.get("DATASETS", {}) or {}
        stereomis_config = datasets_config.get("STEREOMIS", None)
        stir_config = datasets_config.get("STIR", None)
        if stereomis_config is None and stir_config is None:
            self.logger.info(
                "No STEREOMIS or STIR in config; skipping tracking evaluation.",
                context="TEST",
            )
            return

        height = int(self.config.get("IMAGE_HEIGHT", 384))
        width = int(self.config.get("IMAGE_WIDTH", 384))
        grid_size = int(self.config.get("TRACKING_EVAL_GRID_SIZE", 10))
        use_windowed = bool(self.config.get("TRACKING_EVAL_WINDOWED", False))
        window_size = int(self.config.get("TRACKING_WINDOW_SIZE", 16))

        tracking_dir = os.path.join(self.TEST_DIR, "tracking")
        os.makedirs(tracking_dir, exist_ok=True)

        self.matcher.model.eval()
        has_temporal_tracker = hasattr(self, "temporal_tracker") and self.temporal_tracker is not None
        if has_temporal_tracker:
            self.temporal_tracker.refinement_net.eval()
            self.logger.info("Using TemporalTracker for evaluation", context="TEST")

        # Aggregated state: per-dataset lists of per-sequence metrics and
        # one (``dense`` + ``gt_vs_pred``) video slot logged against stable
        # W&B keys so repeated ``run_tests`` calls append to the same panel.
        all_sequence_metrics: List[Dict[str, Any]] = []
        stereomis_rows: List[List[Any]] = []
        stir_rows: List[List[Any]] = []
        video_payload: Dict[str, Any] = {}
        rng = random.Random(int(self.config.get("TRACKING_VAL_VIDEO_SEED", 0)))

        def _log_video_now(key: str, path: Optional[str], fps: int, caption: str) -> None:
            """Log one W&B video immediately to avoid payload memory spikes."""
            if self.wandb is None or not path or not os.path.isfile(path):
                return
            try:
                payload = {
                    key: wandb.Video(path, fps=max(int(fps), 4), format="mp4", caption=caption),
                }
                payload.update(self._wandb_test_eval_step_dict())
                self.wandb.log(payload)
            except Exception as e:
                self.logger.info(f"  Video log failed ({key}): {e}", context="TEST")
        few_use, few_frames, _ = self._tracking_val_fewframes_caps()

        stereomis_stop: Optional[int] = few_frames if few_use else None
        sm_test_cap = self.config.get("TRACKING_STEREOMIS_TEST_MAX_FRAMES", None)
        if sm_test_cap is not None and sm_test_cap != "":
            try:
                sm_test_cap_i = int(sm_test_cap)
                if sm_test_cap_i > 1:
                    stereomis_stop = sm_test_cap_i
            except Exception:
                pass

        stir_stop: Optional[int] = few_frames if few_use else None
        stir_test_cap = self.config.get("TRACKING_STIR_TEST_MAX_FRAMES", None)
        if stir_test_cap is not None and stir_test_cap != "":
            try:
                stir_test_cap_i = int(stir_test_cap)
                if stir_test_cap_i > 1:
                    stir_stop = stir_test_cap_i
            except Exception:
                pass

        stereomis_sequences: List[str] = []
        stereomis_root = ""
        stereomis_fps = 4
        if stereomis_config is not None:
            config_path = stereomis_config.get("PATH", "StereoMIS_Tracking")
            stereomis_root = resolve_dataset_filesystem_path(config_path, "STEREOMIS")
            if stereomis_root is None:
                stereomis_root = os.path.expandvars(os.path.expanduser(str(config_path)))

            all_seq = StereoMISTracking.available_sequences(stereomis_root)
            test_patterns = merge_phase_video_list(
                stereomis_config, "tracking", "TEST_VIDEOS",
            )
            if test_patterns is None or (
                isinstance(test_patterns, list) and len(test_patterns) == 0
            ):
                stereomis_sequences = all_seq
            else:
                stereomis_sequences = select_videos_by_patterns(all_seq, test_patterns)
            stereomis_fps = int(stereomis_config.get("FPS", 4))
            if len(stereomis_sequences) == 0:
                self.logger.info(
                    f"No StereoMIS sequences under {stereomis_root} after TEST_VIDEOS filter.",
                    context="TEST",
                )
            else:
                self.logger.info(
                    f">> STEREOMIS tracking eval: {len(stereomis_sequences)} sequence(s) "
                    f"(windowed={use_windowed}, grid={grid_size}x{grid_size})",
                    context="TEST",
                )
            if stereomis_stop is not None:
                self.logger.info(
                    f"  STEREOMIS test frame cap: stop={stereomis_stop}",
                    context="TEST",
                )

        stereomis_video_seq = (
            rng.choice(stereomis_sequences) if stereomis_sequences else None
        )
        for seq_name in stereomis_sequences:
            self.logger.info(f"  Tracking sequence: {seq_name}", context="TEST")

            try:
                ds = StereoMISTracking(
                    root=stereomis_root,
                    sequence=seq_name,
                    height=height,
                    width=width,
                    start=0,
                    stop=stereomis_stop,
                    step=1,
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

            grid_vis_np: torch.Tensor | None = None
            if has_temporal_tracker:
                with torch.no_grad():
                    tracker_out = self.temporal_tracker.track_long_sequence(
                        grid_pts.unsqueeze(0).to(self.device),
                        _cached_all_frames,
                        window_size=window_size,
                        **self._track_long_sequence_infer_kwargs(),
                    )
                trajectories = tracker_out["tracks"].squeeze(0).permute(1, 0, 2).cpu()  # [T, N, 2]
                grid_vis_logits = tracker_out.get("visibility", None)
                if grid_vis_logits is not None:
                    grid_vis_np = (
                        torch.sigmoid(grid_vis_logits.squeeze(0)) > 0.5
                    ).cpu()  # [N, T]
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

            safe_tag = str(seq_name).replace("/", "_").replace(os.sep, "_")
            is_chosen_video = seq_name == stereomis_video_seq
            video_path = os.path.join(tracking_dir, f"{safe_tag}_stereomis_tracking.mp4")
            if is_chosen_video:
                try:
                    render_tracks_video(
                        dataset=ds,
                        trajectories=trajectories,
                        output_path=video_path,
                        fps=max(stereomis_fps, 4),
                        trail_length=max(5, stereomis_fps),
                        point_radius=3,
                        visibility=grid_vis_np,
                    )
                    self.logger.info(
                        f"  Saved tracking video: {video_path}", context="TEST",
                    )
                except RuntimeError as e:
                    self.logger.info(
                        f"  Video render failed for {seq_name}: {e}", context="TEST",
                    )
                    video_path = None

                if (
                    self.wandb is not None
                    and video_path is not None
                    and os.path.isfile(video_path)
                ):
                    _log_video_now(
                        key="Test/tracking/eval/stereomis/dense_video",
                        path=video_path,
                        fps=stereomis_fps,
                        caption=seq_name,
                    )
            else:
                video_path = None

            seq_metrics = {"sequence": seq_name, "dataset": "STEREOMIS"}
            if ds.tracking_points is not None:
                gt_init_pts = ds.tracking_points[:, 0, :]  # [N_pts, 2]

                if has_temporal_tracker:
                    with torch.no_grad():
                        gt_tracker_out = self.temporal_tracker.track_long_sequence(
                            gt_init_pts.unsqueeze(0).to(self.device),
                            _cached_all_frames,
                            window_size=window_size,
                            **self._track_long_sequence_infer_kwargs(),
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
                if has_temporal_tracker:
                    vis_logits = gt_tracker_out["visibility"].squeeze(0)
                    pred_vis = torch.sigmoid(vis_logits) > 0.5
                else:
                    pred_vis = torch.ones_like(gt_vis)

                results = compute_tap_metrics(
                    pred_tracks, gt_tracks, gt_vis, pred_vis
                )
                seq_metrics.update(results)
                stereomis_rows.append([
                    seq_name,
                    int(gt_tracks.shape[0]),
                    int(gt_tracks.shape[1]),
                    float(results["delta_avg"]),
                    float(results["OA"]),
                    float(results["AJ"]),
                ])

                self.logger.info(
                    f"  {seq_name}: delta_avg={results['delta_avg']:.4f}  "
                    f"OA={results['OA']:.4f}  AJ={results['AJ']:.4f}",
                    context="TEST",
                )

                if is_chosen_video:
                    pixel_errors = (pred_tracks - gt_tracks).norm(dim=-1)  # [N_pts, T_pred]
                    gt_traj_for_video = gt_tracks.permute(1, 0, 2)  # [T_pred, N_pts, 2]
                    cmp_video_path = os.path.join(
                        tracking_dir, f"{safe_tag}_stereomis_gt_vs_pred.mp4",
                    )
                    try:
                        gate_pred = bool(self.config.get(
                            "TRACKING_VAL_COMP_GATE_PREDVIS", False,
                        ))
                        render_comparison_video(
                            dataset=ds,
                            pred_trajectories=eval_traj,
                            gt_trajectories=gt_traj_for_video,
                            output_path=cmp_video_path,
                            fps=max(stereomis_fps, 4),
                            trail_length=max(5, stereomis_fps),
                            point_radius=3,
                            visibility=gt_vis,
                            # Always pass predicted visibility when available
                            # so the video explicitly renders occluded-vs-
                            # visible predictions and the outer confusion ring.
                            pred_visibility=(
                                pred_vis if has_temporal_tracker else None
                            ),
                            errors=pixel_errors,
                            hide_pred_when_occluded=gate_pred,
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
                        _log_video_now(
                            key="Test/tracking/eval/stereomis/gt_vs_pred_video",
                            path=cmp_video_path,
                            fps=stereomis_fps,
                            caption=seq_name,
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

        if stir_config is not None:
            stir_path_cfg = stir_config.get("PATH", "STIR")
            stir_root = resolve_dataset_filesystem_path(stir_path_cfg, "STIR")
            if stir_root is None:
                stir_root = os.path.expandvars(os.path.expanduser(str(stir_path_cfg)))
            col_globs = stir_config.get("COLLECTION_GLOBS", ["*"])
            camera = str(stir_config.get("CAMERA", "left")).lower()
            all_stir = STIRTracking.available_sequences(stir_root, col_globs, camera)
            stir_patterns = merge_phase_video_list(stir_config, "tracking", "TEST_VIDEOS")
            if stir_patterns is None or (
                isinstance(stir_patterns, list) and len(stir_patterns) == 0
            ):
                stir_sequences = all_stir
            else:
                stir_sequences = select_videos_by_patterns(all_stir, stir_patterns)
            stir_fps = int(stir_config.get("FPS", 8))
            if len(stir_sequences) == 0:
                self.logger.info(
                    f"No STIR sequences under {stir_root} after TEST_VIDEOS filter.",
                    context="TEST",
                )
            else:
                self.logger.info(
                    f">> STIR tracking eval: {len(stir_sequences)} sequence(s) "
                    f"(camera={camera}, grid={grid_size}x{grid_size})",
                    context="TEST",
                )
            if stir_stop is not None:
                self.logger.info(
                    f"  STIR test frame cap: stop={stir_stop}",
                    context="TEST",
                )
            stir_video_seq = rng.choice(stir_sequences) if stir_sequences else None
            for seq_name in stir_sequences:
                self.logger.info(f"  STIR sequence: {seq_name}", context="TEST")
                try:
                    ds = STIRTracking(
                        root=stir_root,
                        sequence=seq_name,
                        height=height,
                        width=width,
                        start=0,
                        stop=stir_stop,
                        step=1,
                    )
                except (FileNotFoundError, RuntimeError, OSError) as e:
                    self.logger.info(f"  Skipping STIR {seq_name}: {e}", context="TEST")
                    continue
                if len(ds) < 2:
                    self.logger.info(
                        f"  Skipping STIR {seq_name}: fewer than 2 frames.", context="TEST"
                    )
                    continue
                h, w = ds[0]["image"].shape[1:]
                _cached_all_frames = None
                if has_temporal_tracker:
                    _cached_all_frames = torch.stack(
                        [ds[t]["image"] for t in range(len(ds))], dim=0,
                    ).unsqueeze(0).to(self.device)  # [1, T, 3, H, W]

                # --- GT-anchored tracking: start from the IR-tattoo centers ---
                start_pts = ds.start_points  # [N_start, 2] in processing grid
                end_pts_orig = ds.end_points_orig  # [N_end, 2] in native STIR grid
                h_orig, w_orig = ds.orig_size

                seq_stir_metrics: Dict[str, Any] = {
                    "sequence": seq_name, "dataset": "STIR",
                }
                pred_tracks_gt: Optional[torch.Tensor] = None
                if has_temporal_tracker and start_pts.numel() > 0 and end_pts_orig.numel() > 0:
                    with torch.no_grad():
                        gt_tracker_out = self.temporal_tracker.track_long_sequence(
                            start_pts.unsqueeze(0).to(self.device),
                            _cached_all_frames,
                            window_size=window_size,
                            **self._track_long_sequence_infer_kwargs(),
                        )
                    pred_tracks_gt = gt_tracker_out["tracks"].squeeze(0)  # [N_start, T, 2]
                    _gt_vis_logits_stir = gt_tracker_out.get("visibility", None)
                    pred_vis_gt_stir: torch.Tensor | None = None
                    if _gt_vis_logits_stir is not None:
                        pred_vis_gt_stir = (
                            torch.sigmoid(_gt_vis_logits_stir.squeeze(0)) > 0.5
                        ).cpu()  # [N_start, T]
                    sx = float(w_orig) / max(float(width), 1.0)
                    sy = float(h_orig) / max(float(height), 1.0)
                    pred_end_orig = pred_tracks_gt[:, -1, :].clone()
                    pred_end_orig[..., 0] *= sx
                    pred_end_orig[..., 1] *= sy
                    metric = compute_stir_endpoint_metrics(
                        pred_end_orig.cpu(), end_pts_orig,
                        thresholds=STIR_THRESHOLDS_PX,
                    )
                    stir_rows.append([
                        seq_name,
                        int(metric["num_query_points"]),
                        int(metric["num_gt_points"]),
                        float(metric["delta_avg"]),
                        float(metric["mean_dist_px"]),
                        float(metric["median_dist_px"]),
                        *[float(metric[f"acc_{int(th)}px"]) for th in STIR_THRESHOLDS_PX],
                    ])
                    seq_stir_metrics.update({
                        k: v for k, v in metric.items()
                        if isinstance(v, float) or isinstance(v, int)
                    })
                    self.logger.info(
                        f"  {seq_name}: STIR δavg={metric['delta_avg']:.4f} "
                        f"mean={metric['mean_dist_px']:.2f}px "
                        f"N={int(metric['num_query_points'])}",
                        context="TEST",
                    )

                # --- Videos (only for the chosen sequence) ---
                is_chosen_video = seq_name == stir_video_seq
                if is_chosen_video:
                    grid_pts = make_grid_points(h, w, grid_h=grid_size, grid_w=grid_size)
                    grid_vis_np: torch.Tensor | None = None
                    if has_temporal_tracker:
                        with torch.no_grad():
                            tracker_out = self.temporal_tracker.track_long_sequence(
                                grid_pts.unsqueeze(0).to(self.device),
                                _cached_all_frames,
                                window_size=window_size,
                                **self._track_long_sequence_infer_kwargs(),
                            )
                        trajectories = tracker_out["tracks"].squeeze(0).permute(1, 0, 2).cpu()
                        grid_vis_logits = tracker_out.get("visibility", None)
                        if grid_vis_logits is not None:
                            grid_vis_np = (
                                torch.sigmoid(grid_vis_logits.squeeze(0)) > 0.5
                            ).cpu()  # [N, T]
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
                        )
                    safe_tag = str(seq_name).replace("/", "_").replace(os.sep, "_")
                    video_path = os.path.join(tracking_dir, f"{safe_tag}_stir_tracking.mp4")
                    try:
                        render_tracks_video(
                            dataset=ds,
                            trajectories=trajectories,
                            output_path=video_path,
                            fps=max(stir_fps, 4),
                            trail_length=max(5, stir_fps),
                            point_radius=3,
                            visibility=grid_vis_np,
                        )
                    except RuntimeError as e:
                        self.logger.info(
                            f"  STIR video render failed for {seq_name}: {e}", context="TEST",
                        )
                        video_path = None
                    if self.wandb is not None and video_path and os.path.isfile(video_path):
                        _log_video_now(
                            key="Test/tracking/eval/stir/dense_video",
                            path=video_path,
                            fps=stir_fps,
                            caption=seq_name,
                        )

                    if pred_tracks_gt is not None:
                        T_total = int(pred_tracks_gt.shape[1])
                        end_pts_proc = ds.end_points.to(pred_tracks_gt.device)
                        if end_pts_proc.numel() > 0:
                            pred_end_proc = pred_tracks_gt[:, -1, :]
                            dmat = torch.cdist(
                                pred_end_proc.unsqueeze(0),
                                end_pts_proc.unsqueeze(0),
                            ).squeeze(0)
                            matched_end = end_pts_proc[dmat.argmin(dim=1)]
                        else:
                            matched_end = pred_tracks_gt[:, -1, :].clone()
                        alpha = torch.linspace(
                            0.0, 1.0, T_total, device=pred_tracks_gt.device,
                        ).view(1, T_total, 1)
                        gt_traj_proc = (1.0 - alpha) * start_pts.to(
                            pred_tracks_gt.device
                        ).unsqueeze(1) + alpha * matched_end.unsqueeze(1)
                        gt_vis = torch.zeros(
                            pred_tracks_gt.shape[0], T_total, dtype=torch.bool,
                        )
                        gt_vis[:, 0] = True
                        gt_vis[:, -1] = True
                        pred_errors = (
                            pred_tracks_gt.cpu() - gt_traj_proc.cpu()
                        ).norm(dim=-1)
                        cmp_video_path = os.path.join(
                            tracking_dir, f"{safe_tag}_stir_gt_vs_pred.mp4",
                        )
                        try:
                            render_comparison_video(
                                dataset=ds,
                                pred_trajectories=pred_tracks_gt.permute(1, 0, 2).cpu(),
                                gt_trajectories=gt_traj_proc.permute(1, 0, 2).cpu(),
                                output_path=cmp_video_path,
                                fps=max(stir_fps, 4),
                                trail_length=max(5, stir_fps),
                                point_radius=3,
                                visibility=gt_vis,
                                pred_visibility=pred_vis_gt_stir,
                                errors=pred_errors,
                                gate_prediction_on_gt_vis=False,
                            )
                        except RuntimeError as e:
                            self.logger.info(
                                f"  STIR comparison video render failed for {seq_name}: {e}",
                                context="TEST",
                            )
                            cmp_video_path = None
                        if (
                            self.wandb is not None
                            and cmp_video_path is not None
                            and os.path.isfile(cmp_video_path)
                        ):
                            _log_video_now(
                                key="Test/tracking/eval/stir/gt_vs_pred_video",
                                path=cmp_video_path,
                                fps=stir_fps,
                                caption=seq_name,
                            )

                all_sequence_metrics.append(seq_stir_metrics)
                del ds
                if _cached_all_frames is not None:
                    del _cached_all_frames
                torch.cuda.empty_cache()

        # ------ Aggregate, log once with stable per-dataset keys ------
        summary_payload: Dict[str, Any] = dict(video_payload)
        if stereomis_rows:
            delta = [r[3] for r in stereomis_rows]
            oa = [r[4] for r in stereomis_rows]
            aj = [r[5] for r in stereomis_rows]
            prefix_sm = "Test/tracking/eval/stereomis"
            summary_payload[f"{prefix_sm}/delta_avg_mean"] = float(np.nanmean(delta))
            summary_payload[f"{prefix_sm}/OA_mean"] = float(np.nanmean(oa))
            summary_payload[f"{prefix_sm}/AJ_mean"] = float(np.nanmean(aj))
            summary_payload[f"{prefix_sm}/num_sequences"] = int(len(stereomis_rows))
            self.logger.info(
                f">> STEREOMIS test ({len(stereomis_rows)} seqs): "
                f"δavg={summary_payload[f'{prefix_sm}/delta_avg_mean']:.4f} "
                f"OA={summary_payload[f'{prefix_sm}/OA_mean']:.4f} "
                f"AJ={summary_payload[f'{prefix_sm}/AJ_mean']:.4f}",
                context="TEST",
            )
            if self.wandb is not None:
                summary_payload[f"{prefix_sm}/per_sequence"] = wandb.Table(
                    columns=["sequence", "num_points", "num_frames", "delta_avg", "OA", "AJ"],
                    data=stereomis_rows,
                )

        if stir_rows:
            prefix_st = "Test/tracking/eval/stir"
            delta = [r[3] for r in stir_rows]
            mean_d = [r[4] for r in stir_rows]
            med_d = [r[5] for r in stir_rows]
            summary_payload[f"{prefix_st}/delta_avg_mean"] = float(np.nanmean(delta))
            summary_payload[f"{prefix_st}/mean_dist_px_mean"] = float(np.nanmean(mean_d))
            summary_payload[f"{prefix_st}/median_dist_px_mean"] = float(np.nanmean(med_d))
            summary_payload[f"{prefix_st}/num_sequences"] = int(len(stir_rows))
            for j, th in enumerate(STIR_THRESHOLDS_PX):
                vals = [r[6 + j] for r in stir_rows]
                summary_payload[f"{prefix_st}/acc_{int(th)}px_mean"] = float(np.nanmean(vals))
            self.logger.info(
                f">> STIR test ({len(stir_rows)} seqs): "
                f"δavg={summary_payload[f'{prefix_st}/delta_avg_mean']:.4f} "
                f"mean_dist={summary_payload[f'{prefix_st}/mean_dist_px_mean']:.2f}px",
                context="TEST",
            )
            if self.wandb is not None:
                summary_payload[f"{prefix_st}/per_sequence"] = wandb.Table(
                    columns=[
                        "sequence", "num_query", "num_gt",
                        "delta_avg", "mean_dist_px", "median_dist_px",
                        *[f"acc_{int(th)}px" for th in STIR_THRESHOLDS_PX],
                    ],
                    data=stir_rows,
                )

        if self.wandb is not None and summary_payload:
            try:
                summary_payload.update(self._wandb_test_eval_step_dict())
                self.wandb.log(summary_payload)
            except Exception as e:
                self.logger.info(f"  Test-tracking wandb summary log failed: {e}", context="TEST")

        if all_sequence_metrics:
            tracking_csv_path = os.path.join(tracking_dir, "tracking_metrics.csv")
            pd.DataFrame(all_sequence_metrics).to_csv(tracking_csv_path, index=False)
            self.logger.info(
                f"  Tracking metrics CSV: {os.path.abspath(tracking_csv_path)}",
                context="TEST",
            )

    def log_tests(self) -> None:
        """
        Logs the test metrics to Weights and Biases.
        """

        self.logger.info(">> TEST REPORT", context="TEST")
        test_df = self.metrics["Test"]
        if len(test_df.columns) == 0 or len(test_df) == 0:
            self.logger.info(
                "No pairwise Test-phase metrics (empty frame; e.g. PHASE=tracking skips "
                "``run_epoch(Test)``). Tracking summaries are logged separately.",
                context="TEST",
            )
        else:
            test_describe = test_df.describe().to_string()
            self.logger.info("\n" + test_describe, context="TEST")
            del test_describe
        test_df.to_csv(self.TEST_DIR + "/test_metrics.csv")
        if self.wandb is not None and len(test_df.columns) > 0 and len(test_df) > 0:
            test_table = wandb.Table(dataframe=test_df)
            self.wandb.log({"Test/Summary": test_table, **self._wandb_test_eval_step_dict()})
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
        if not is_main_process():
            return

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

        diagnostics = getattr(unwrap_model(self.matcher.model), "latest_diagnostics", {})
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

            # Per-layer mean gate weight across batch + patch dims, keyed by
            # the absolute DINOv3 layer index:
            #   layer_weights: [B_all, L, N]  ->  per_layer_mean: [L]
            per_layer_mean = layer_weights.mean(dim=(0, 2))  # [L]
            layer_index_source = None
            for diag in (source_diag, target_diag):
                if diag is not None and diag.get("layer_indices") is not None:
                    layer_index_source = diag["layer_indices"]
                    break
            if layer_index_source is not None:
                layer_id_list = [int(i) for i in layer_index_source.tolist()]
            else:
                layer_id_list = list(range(num_layers))
            per_layer_values = per_layer_mean.detach().cpu().tolist()  # length L
            # Python loop over at most L (=24) metric names is unavoidable:
            # the tensor reductions above are already vectorized; this is only
            # populating a scalar metric dict.
            for layer_id, val in zip(layer_id_list, per_layer_values):
                metrics[f"gate/per_layer/layer_{layer_id:02d}"] = val
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

        metrics["fused/pos_similarity"] = fused_pos.mean().item()
        metrics["fused/margin"] = fused_margin.mean().item()
        metrics["fused/gain"] = (fused_margin.mean() - raw_margin.mean()).item()

        triplet_confidence = triplets_dict.get("triplet_confidence")
        if triplet_confidence is not None and triplet_confidence.numel() > 0:
            triplet_confidence = triplet_confidence.float()
            metrics["confidence/mean"] = triplet_confidence.mean().item()

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
        diagnostics = getattr(unwrap_model(self.matcher.model), "latest_diagnostics", {})
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
                unwrap_model(self.matcher.model).fromArtifact(pth_namestring=resolved)
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
        if is_ddp_enabled(self.config):
            lr = int(self.config.get("LOCAL_RANK", 0))
            self.temporal_tracker.refinement_net = nn.parallel.DistributedDataParallel(
                self.temporal_tracker.refinement_net,
                device_ids=[lr],
                output_device=lr,
                find_unused_parameters=ddp_find_unused_parameters(self.config),
                static_graph=ddp_static_graph(self.config),
            )

        lr_temporal = float(
            self.config.get(
                "LR_TEMPORAL_REFINEMENT",
                self.config.get("LEARNING_RATE", 1e-4),
            )
        )
        tracking_opt_name = str(self.config.get("OPTIMIZER", "Adam"))
        tracking_opt_cls = getattr(optimization, tracking_opt_name)
        self.tracking_optimizer = tracking_opt_cls(
            self.temporal_tracker.refinement_net.parameters(),
            lr=lr_temporal,
            weight_decay=float(self.config.get("WEIGHT_DECAY", 0)),
        )

        # The descriptor-stage LR schedulers (``self.LRscheduler`` /
        # ``self.LRschedulerPlateau``) are bound to ``self.optimizer`` (the
        # frozen matcher's optimizer) and therefore never affected the
        # refinement net. Build a dedicated cosine scheduler on the tracking
        # optimizer so the LR actually decays during PHASE=tracking.
        epochs = max(1, int(self.config.get("EPOCHS", 1)))
        eta_min = float(
            self.config.get(
                "LR_TEMPORAL_REFINEMENT_MIN",
                lr_temporal * 1e-2,
            )
        )
        warmup_ep = max(0, int(self.config.get("TRACKING_LR_WARMUP_EPOCHS", 0)))
        if warmup_ep > 0:
            w_start = float(self.config.get("TRACKING_LR_WARMUP_START_FACTOR", 0.01))
            warmup = torch.optim.lr_scheduler.LinearLR(
                self.tracking_optimizer,
                start_factor=w_start,
                end_factor=1.0,
                total_iters=warmup_ep,
            )
            cosine_T = max(1, epochs - warmup_ep)
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.tracking_optimizer,
                T_max=cosine_T,
                eta_min=eta_min,
            )
            self.tracking_lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.tracking_optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_ep],
            )
        else:
            self.tracking_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.tracking_optimizer,
                T_max=epochs,
                eta_min=eta_min,
            )

        self._pseudo_cycle_released = False
        self._pseudo_cycle_below_streak = 0

        # Must not use ``self.earlystopping`` here: that writes ``{runname}_checkpoint.pth``,
        # which ``fromArtifact`` / ``reinstantiate_model_from_checkpoint`` treat as the full
        # matcher. Save refinement-only checkpoints under a distinct basename instead.
        self.tracking_earlystopping = initialize.earlystopping(
            self.earlystopping_patience,
            self.MODELS_DIR,
            f"{self.runname}_tracking_refinement",
        )

        from gatetracker.data import SequenceWindowDataset

        window_size = int(self.config.get("TRACKING_SEQUENCE_LENGTH", 8))
        default_ws_stride = max(1, window_size // 2)
        train_ws = self.config.get("TRACKING_WINDOW_TRAIN_STRIDE")
        train_ws_stride = (
            max(1, int(train_ws)) if train_ws is not None else default_ws_stride
        )
        self._tracking_datasets = {}
        for phase_name in ("Training", "Validation"):
            base_ds = self.dataset[phase_name]
            try:
                stride = (
                    train_ws_stride
                    if phase_name == "Training"
                    else default_ws_stride
                )
                seq_ds = SequenceWindowDataset(
                    base_ds,
                    window_size=window_size,
                    stride=stride,
                    mode="train" if phase_name == "Training" else "eval",
                )
                self._tracking_datasets[phase_name] = seq_ds
                self.logger.info(
                    f"Built {phase_name} sequence dataset: {len(seq_ds)} windows of "
                    f"{window_size} frames (start stride={stride})",
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
        if is_main_process():
            torch.save(
                unwrap_model(self.temporal_tracker.refinement_net).state_dict(),
                tracking_ckpt_path,
            )
            self.logger.info(f"Saved tracking refinement network: {tracking_ckpt_path}", context="SAVE")
        self.logger.info("TRACKING TRAINING COMPLETE", context="SAVE")

    def _sched_cycle_loss_weight(self, epoch: int, cfg) -> float:
        """Epoch schedule for cycle loss weight (piecewise milestones or linear ramp)."""
        raw = cfg.get("TEMPORAL_CYCLE_LOSS_MILESTONES")
        milestones = None
        if raw is not None:
            milestones = raw.get("value", raw) if isinstance(raw, dict) else raw
        if milestones:
            return _piecewise_linear_epochs(epoch, milestones)
        cyc_s = float(cfg.get("TEMPORAL_CYCLE_LOSS_WEIGHT", 1.0))
        cyc_e = float(cfg.get("TEMPORAL_CYCLE_LOSS_WEIGHT_FINAL", cyc_s))
        cyc_r = int(cfg.get("TEMPORAL_CYCLE_LOSS_RAMP_EPOCHS", 0))
        return _linear_ramp(epoch, cyc_s, cyc_e, cyc_r)

    def _sched_pseudo_sup_lambda(self, epoch: int, cfg) -> float:
        """Blend weight ``lambda`` for pseudo-GT supervised vs self-sup (decoupled from cycle ramp)."""
        lam_max = float(cfg.get("PSEUDO_GT_SUP_LAMBDA_MAX", 0.0))
        if lam_max <= 0.0:
            return 0.0
        ramp_ep = int(
            cfg.get(
                "PSEUDO_GT_SUP_LAMBDA_RAMP_EPOCHS",
                cfg.get("PSEUDO_GT_CURRICULUM_EPOCHS", 10),
            )
        )
        ramp_ep = max(1, ramp_ep)
        lam_uncapped = lam_max * min(1.0, float(epoch + 1) / float(ramp_ep))
        cap = float(cfg.get("PSEUDO_GT_SUP_LAMBDA_CAP_UNTIL_CYCLE_RELEASE", 0.05))
        thresh = cfg.get("PSEUDO_GT_CYCLE_LOSS_RELEASE_THRESH", None)
        hold_epochs = int(cfg.get("PSEUDO_GT_SUP_LAMBDA_PLATEAU_HOLD_EPOCHS", 0))
        released = bool(getattr(self, "_pseudo_cycle_released", False))
        if thresh is not None:
            if not released:
                return float(min(cap, lam_uncapped))
            return float(lam_uncapped)
        if hold_epochs > 0 and int(epoch) < hold_epochs:
            return float(min(cap, lam_uncapped))
        return float(lam_uncapped)

    def _tracking_update_pseudo_lambda_release_gate(
        self, mean_loss_cycle: float, cfg,
    ) -> None:
        """Advance loss-gated release for pseudo ``lambda`` (DDP-safe mean cycle loss)."""
        thresh = cfg.get("PSEUDO_GT_CYCLE_LOSS_RELEASE_THRESH", None)
        if thresh is None or getattr(self, "_pseudo_cycle_released", False):
            return
        thresh_f = float(thresh)
        patience = max(1, int(cfg.get("PSEUDO_GT_CYCLE_LOSS_RELEASE_PATIENCE", 2)))
        if not math.isfinite(mean_loss_cycle):
            return
        if mean_loss_cycle < thresh_f:
            self._pseudo_cycle_below_streak = int(
                getattr(self, "_pseudo_cycle_below_streak", 0),
            ) + 1
            if self._pseudo_cycle_below_streak >= patience:
                self._pseudo_cycle_released = True
        else:
            self._pseudo_cycle_below_streak = 0

    def _run_tracking_epoch(self, phase: str = "Training", epoch: int = 0) -> str:
        """Execute one tracking training/validation epoch.

        Args:
            phase: "Training" or "Validation".
            epoch: Current epoch index.

        Returns:
            Status string.
        """
        TRAINING = phase == "Training"
        self.current_epoch = int(epoch)
        seq_ds = self._tracking_datasets.get(phase)
        if seq_ds is None:
            # Still run StereoMIS GT / video eval on validation epochs so GT vs pred
            # MP4s are produced even when SequenceWindowDataset(Validation) failed to build.
            if not TRAINING:
                self._tracking_validation_tracking_epoch(epoch)
            return "SKIP"

        if TRAINING:
            self.temporal_tracker.refinement_net.train()
        else:
            self.temporal_tracker.refinement_net.eval()

        def tracking_collate_fn(batch):
            frames = torch.stack([b["frames"] for b in batch], dim=0)  # [B, T, 3, H, W]
            return {"frames": frames}

        from torch.utils.data.distributed import DistributedSampler

        tracking_batch_size = int(self.config.get("TRACKING_BATCH_SIZE", max(1, self.batch_size)))
        num_workers = int(self.config.get("TRACKING_WORKERS", max(4, self.config.WORKERS)))
        if is_ddp_enabled(self.config):
            _samp = DistributedSampler(
                seq_ds,
                num_replicas=int(self.config.get("WORLD_SIZE", 1)),
                rank=int(self.config.get("RANK", 0)),
                shuffle=TRAINING,
                drop_last=True,
            )
            _samp.set_epoch(int(epoch))
            dataloader = torch.utils.data.DataLoader(
                seq_ds,
                batch_size=tracking_batch_size,
                num_workers=num_workers,
                drop_last=True,
                collate_fn=tracking_collate_fn,
                shuffle=False,
                sampler=_samp,
                pin_memory=True,
                prefetch_factor=2 if num_workers > 0 else None,
                persistent_workers=num_workers > 0,
            )
        else:
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
                self._tracking_validation_tracking_epoch(epoch)
            return "EMPTY"

        cfg_sched = self.config
        mix_start = float(cfg_sched.get("PSEUDO_GT_MIX", 0.8))
        mix_end = float(cfg_sched.get("PSEUDO_GT_MIX_MIN", mix_start))
        mix_ramp = int(cfg_sched.get("PSEUDO_GT_MIX_RAMP_EPOCHS", 0))
        pseudo_gt_mix_now = _linear_ramp(epoch, mix_start, mix_end, mix_ramp)

        cycle_w_now = self._sched_cycle_loss_weight(int(epoch), cfg_sched)

        sm_s = float(cfg_sched.get("TEMPORAL_SMOOTHNESS_LOSS_WEIGHT", 0.5))
        sm_e = float(cfg_sched.get("TEMPORAL_SMOOTHNESS_LOSS_WEIGHT_FINAL", sm_s))
        sm_r = int(cfg_sched.get("TEMPORAL_SMOOTHNESS_LOSS_RAMP_EPOCHS", 0))
        smooth_w_now = _linear_ramp(epoch, sm_s, sm_e, sm_r)

        cfg_sched["_SCHED_PSEUDO_GT_MIX"] = pseudo_gt_mix_now
        cfg_sched["_SCHED_CYCLE_LOSS_WEIGHT"] = cycle_w_now
        cfg_sched["_SCHED_SMOOTHNESS_LOSS_WEIGHT"] = smooth_w_now
        cfg_sched["_SCHED_PSEUDO_SUP_LAMBDA"] = self._sched_pseudo_sup_lambda(
            int(epoch), cfg_sched,
        )

        num_query_pts = int(self.config.get("TRACKING_NUM_QUERY_POINTS", 64))
        epoch_losses = []
        # Epoch-level aggregates so W&B run summary is not the last-batch value
        # (pseudo-GT activation is a Bernoulli(mix) per step; any single batch
        # can look inactive even when the rate is mix).
        epoch_agg = {
            "pseudo_gt_active": 0.0,
            "pseudo_lambda": 0.0,
            "grad_norm": 0.0,
            "grad_norm_clipped": 0.0,
            "n_steps_skipped": 0,
            "loss_cycle_sum": 0.0,
            "loss_cycle_count": 0,
        }
        epoch_agg_count = 0

        grad_clip_max = float(
            self.config.get(
                "TRACKING_GRAD_CLIP_MAX_NORM",
                optimization.GRAD_CLIP_MAX_NORM,
            )
        )

        ctx = nullcontext() if TRAINING else torch.no_grad()
        with ctx:
            for batch_idx, batch in enumerate(dataloader):
                if (
                    TRAINING
                    and os.environ.get("GATETRACKER_CRASH_TEST", "").lower()
                    in ("1", "true")
                    and int(batch_idx) == 0
                    and int(epoch) == 0
                ):
                    raise RuntimeError("GATETRACKER_CRASH_TEST")

                frames = batch["frames"].to(self.device)  # [B, T, 3, H, W]
                self.global_step = int(epoch) * len(dataloader) + int(batch_idx)

                loss_dict = self.temporal_tracker.training_step(
                    frames,
                    self.config,
                    num_query_points=num_query_pts,
                    batch=batch,
                    geometry_pipeline=self.geometryPipeline,
                    epoch=epoch,
                )

                loss_total = loss_dict["loss_total"]
                metrics_self_sup = loss_dict.get("metrics_self_sup", {}) or {}
                metrics_pseudo_gt = loss_dict.get("metrics_pseudo_gt", {}) or {}
                pga = float(metrics_pseudo_gt.get("pseudo_gt_active", 0.0))
                self_sup_total_f = metrics_self_sup.get("loss_self_sup_total")
                pseudo_sup_total_f = metrics_pseudo_gt.get("loss_sup_total")

                grad_norm_val = 0.0
                grad_norm_clipped_val = 0.0
                weight_norm_val = 0.0
                step_skipped = False
                if TRAINING:
                    if not torch.isfinite(loss_total):
                        self.logger.warning(
                            f"Non-finite total loss at epoch {epoch}, step {self.global_step}: "
                            f"total={loss_total.item()}, "
                            f"self_sup={self_sup_total_f}, "
                            f"pseudo_sup={pseudo_sup_total_f}, "
                            f"pseudo_gt_active={pga > 0.5}"
                        )
                        self.tracking_optimizer.zero_grad(set_to_none=True)
                        if self.wandb is not None:
                            self.wandb.log(
                                {f"{phase}/tracking/optim/StepSkipped": 1},
                                commit=False,
                            )
                        epoch_agg["n_steps_skipped"] += 1
                        continue

                    loss_total.backward()
                    params = list(
                        self.temporal_tracker.refinement_net.parameters()
                    )
                    grad_norm_val, weight_norm_val = (
                        optimization.get_grad_and_weight_norms(params)
                    )
                    # clip_grad_norm_ returns the pre-clip total grad norm
                    # (for backward compatibility with the matching loop's
                    # convention, we log it as ``GradNormClipped`` → final
                    # post-clip norm is min(grad_norm, max_norm)).
                    pre_clip = torch.nn.utils.clip_grad_norm_(
                        params,
                        max_norm=grad_clip_max,
                    )
                    pre_clip_val = (
                        float(pre_clip.detach().cpu())
                        if isinstance(pre_clip, torch.Tensor)
                        else float(pre_clip)
                    )
                    if not math.isfinite(pre_clip_val):
                        # Non-finite grads → skip optimizer update.
                        self.tracking_optimizer.zero_grad(set_to_none=True)
                        step_skipped = True
                    else:
                        grad_norm_clipped_val = min(pre_clip_val, grad_clip_max)
                        self.tracking_optimizer.step()
                        self.tracking_optimizer.zero_grad(set_to_none=True)

                metrics = self._tracking_step_metrics_for_log(phase, loss_dict)
                metrics[f"Step/{'epoch' if phase == 'Training' else 'val_epoch'}"] = epoch
                metrics["Loss"] = loss_total.item()
                metrics["HyperParameters/pseudo_gt_mix_now"] = float(pseudo_gt_mix_now)
                metrics["HyperParameters/cycle_loss_weight_now"] = float(cycle_w_now)
                metrics["HyperParameters/smoothness_loss_weight_now"] = float(
                    smooth_w_now
                )
                metrics["HyperParameters/pseudo_sup_lambda_sched"] = float(
                    cfg_sched.get("_SCHED_PSEUDO_SUP_LAMBDA", 0.0)
                )
                metrics["HyperParameters/pseudo_cycle_released"] = float(
                    1.0 if getattr(self, "_pseudo_cycle_released", False) else 0.0
                )
                if pga > 0.5:
                    metrics[f"{phase}/loss/Loss_pseudo"] = float(loss_total.item())
                    if self_sup_total_f is not None:
                        metrics[
                            f"{phase}/tracking/self_sup/loss_self_sup_total_pseudo"
                        ] = float(self_sup_total_f)
                else:
                    metrics[f"{phase}/loss/Loss_real"] = float(loss_total.item())
                    if self_sup_total_f is not None:
                        metrics[
                            f"{phase}/tracking/self_sup/loss_self_sup_total_real"
                        ] = float(self_sup_total_f)
                metrics[f"{phase}/optim/LR_temporal"] = float(
                    self.tracking_optimizer.param_groups[0]["lr"]
                )
                if TRAINING:
                    metrics[f"{phase}/tracking/optim/GradNorm"] = grad_norm_val
                    metrics[f"{phase}/tracking/optim/GradNormClipped"] = (
                        grad_norm_clipped_val
                    )
                    metrics[f"{phase}/tracking/optim/WeightNorm"] = weight_norm_val
                    metrics[f"{phase}/tracking/optim/StepSkipped"] = float(
                        step_skipped
                    )
                # Per-epoch accumulators (pseudo_gt + grad stats).
                pga_step = float(
                    loss_dict.get("metrics_pseudo_gt", {}).get("pseudo_gt_active", 0.0)
                )
                pl_step = float(
                    loss_dict.get("metrics_pseudo_gt", {}).get("pseudo_lambda", 0.0)
                )
                epoch_agg["pseudo_gt_active"] += pga_step
                epoch_agg["pseudo_lambda"] += pl_step
                epoch_agg["grad_norm"] += grad_norm_val
                epoch_agg["grad_norm_clipped"] += grad_norm_clipped_val
                epoch_agg["n_steps_skipped"] += int(step_skipped)
                lc = loss_dict.get("metrics_self_sup", {}).get("loss_cycle", None)
                if lc is not None and math.isfinite(float(lc)):
                    epoch_agg["loss_cycle_sum"] += float(lc)
                    epoch_agg["loss_cycle_count"] += 1
                epoch_agg_count += 1

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

                del frames, loss_dict, loss_total
                torch.cuda.empty_cache()

        if TRAINING and float(epoch_agg.get("loss_cycle_count", 0)) > 0:
            lc_sum = float(epoch_agg["loss_cycle_sum"])
            lc_cnt = float(epoch_agg["loss_cycle_count"])
            if is_ddp_enabled(self.config):
                lc_sum, lc_cnt = all_reduce_sum_scalars(lc_sum, lc_cnt, self.device)
            mean_lc = lc_sum / max(lc_cnt, 1.0)
            self._tracking_update_pseudo_lambda_release_gate(mean_lc, cfg_sched)

        # DDP collective BEFORE any rank-0-only work. If we synced AFTER
        # ``_tracking_validation_tracking_epoch`` (which only runs on rank 0
        # and can take many minutes for StereoMIS + STIR video rendering),
        # non-main ranks would sit at the 2-element ALLREDUCE below and trip
        # the NCCL watchdog. Doing the reduce first keeps all ranks in
        # lock-step on collectives, and a ``barrier()`` after the main-only
        # block resynchronises before the next epoch. Also made
        # unconditional on ``epoch_losses`` so an empty-epoch rank cannot
        # silently skip the collective and desync.
        _sum = float(np.sum(epoch_losses)) if epoch_losses else 0.0
        _cnt = float(len(epoch_losses))
        if is_ddp_enabled(self.config):
            _sum, _cnt = all_reduce_sum_scalars(_sum, _cnt, self.device)
        mean_loss = _sum / max(_cnt, 1.0)

        if not TRAINING:
            self._tracking_validation_tracking_epoch(epoch)
            if is_ddp_enabled(self.config):
                barrier()

        if _cnt > 0:
            self.logger.info(
                f">> {phase} epoch {epoch+1} mean loss: {mean_loss:.4f}",
                context="TRACKING",
            )
            if self.wandb is not None:
                payload_epoch = {
                    f"{phase}/tracking/epoch_loss": mean_loss,
                    **self._wandb_epoch_axis_dict(epoch),
                }
                if epoch_agg_count > 0:
                    inv = 1.0 / float(epoch_agg_count)
                    payload_epoch.update({
                        f"{phase}/tracking/pseudo_gt/epoch/pseudo_gt_active_mean": (
                            epoch_agg["pseudo_gt_active"] * inv
                        ),
                        f"{phase}/tracking/pseudo_gt/epoch/pseudo_lambda_mean": (
                            epoch_agg["pseudo_lambda"] * inv
                        ),
                    })
                    if TRAINING:
                        payload_epoch.update({
                            f"{phase}/tracking/optim/epoch/GradNorm_mean": (
                                epoch_agg["grad_norm"] * inv
                            ),
                            f"{phase}/tracking/optim/epoch/GradNormClipped_mean": (
                                epoch_agg["grad_norm_clipped"] * inv
                            ),
                            f"{phase}/tracking/optim/epoch/StepsSkipped": float(
                                epoch_agg["n_steps_skipped"]
                            ),
                        })
                self.wandb.log(payload_epoch)

            if not TRAINING:
                self.step["epoch"] += 1
                # Step the tracking-optimizer scheduler at epoch granularity.
                # ``self.LRschedulerPlateau`` is bound to the descriptor
                # optimizer (frozen here) and is intentionally not stepped.
                if getattr(self, "tracking_lr_scheduler", None) is not None:
                    self.tracking_lr_scheduler.step()
                should_stop = self.tracking_earlystopping(
                    mean_loss, self.temporal_tracker.refinement_net, epoch,
                )
                if should_stop:
                    self.logger.info(
                        ">> [EARLYSTOPPING]: Patience reached for tracking training",
                        context="TRACKING",
                    )
                    return "EARLYSTOP"

        return "COMPLETED"

    def _tracking_step_metrics_for_log(
        self, phase: str, loss_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build W&B / CSV keys for temporal training: separate self-sup vs pseudo-GT."""
        base = f"{phase}/tracking"
        out: Dict[str, Any] = {}
        for k, v in loss_dict.get("metrics_self_sup", {}).items():
            if v is None:
                continue
            out[f"{base}/self_sup/{k}"] = v
        for k, v in loss_dict.get("metrics_pseudo_gt", {}).items():
            if v is None:
                continue
            out[f"{base}/pseudo_gt/{k}"] = v
        return out

    def _tracking_val_fewframes_caps(self) -> Tuple[bool, Optional[int], int]:
        """When ``FEWFRAMES`` is true (``--boot``), cap external val clips.

        Full STIR / StereoMIS sequences can be hundreds of frames; stacking
        ``[1, T, 3, H, W]`` on GPU for every ``VAL_VIDEOS`` clip exhausts VRAM
        and the OS OOM-killer reports ``Killed``. Boot smoke-tests only need a
        short prefix of a couple of clips.

        Returns:
            use_caps: apply sequence / frame limits.
            max_frames: ``stop`` for dataset constructors (``None`` = full clip).
            max_sequences: max number of validation clips to iterate.
        """
        if not bool(self.config.get("FEWFRAMES")):
            return False, None, 1_000_000
        win = int(
            self.config.get(
                "TRACKING_SEQUENCE_LENGTH",
                self.config.get("TRACKING_WINDOW_SIZE", 16),
            )
        )
        # [1, T, 3, H, W] on GPU — keep T modest (OOM-safe); at least 2× window.
        max_frames = max(win * 2, 48)
        max_sequences = 2
        return True, max_frames, max_sequences

    def _tracking_validation_tracking_epoch(self, epoch: int) -> None:
        """StereoMIS + STIR val metrics/videos and pseudo novel-view clip (main rank)."""
        if not hasattr(self, "temporal_tracker") or self.temporal_tracker is None:
            return
        if is_ddp_enabled(self.config) and not is_main_process():
            return
        self._stereomis_validation_epoch(epoch)
        self._stir_validation_epoch(epoch)
        self._tracking_validate_pseudo_novelview(epoch)

    def _stereomis_validation_epoch(self, epoch: int) -> None:
        """StereoMIS val: aggregated TAP metrics + one comparison video.

        All W&B logs use **stable keys** (``Validation/tracking/eval/stereomis/...``)
        with no sequence name inside the key, so each epoch appends to the same
        panel at ``step/epoch``. Per-sequence metrics are emitted as a
        ``wandb.Table`` (one row per clip) once per epoch.
        """
        if not bool(self.config.get("TRACKING_STEREOMIS_VAL", True)):
            return
        import random
        from dataset.loader_phase import merge_phase_video_list
        from dataset.utils import select_videos_by_patterns
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
            return

        config_path = stereomis_config.get("PATH", "StereoMIS_Tracking")
        root = resolve_dataset_filesystem_path(config_path, "STEREOMIS")
        if root is None:
            root = os.path.expandvars(os.path.expanduser(str(config_path)))

        all_seq = StereoMISTracking.available_sequences(root)
        val_patterns = merge_phase_video_list(stereomis_config, "tracking", "VAL_VIDEOS")
        if val_patterns is None or (isinstance(val_patterns, list) and len(val_patterns) == 0):
            val_seqs = all_seq
        else:
            val_seqs = select_videos_by_patterns(all_seq, val_patterns)
        if not val_seqs:
            self.logger.info("StereoMIS val: no sequences after VAL_VIDEOS filter.", context="TRACKING")
            return

        few_use, few_frames, few_max_seq = self._tracking_val_fewframes_caps()
        if few_use:
            val_seqs = val_seqs[:few_max_seq]
            self.logger.info(
                f"StereoMIS val: FEWFRAMES cap → {len(val_seqs)} seq(s), "
                f"max_frames={few_frames}",
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
        if few_use and stop is None:
            stop = few_frames

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
        out_dir = os.path.join(self.RUN_DIR, "tracking_val")
        os.makedirs(out_dir, exist_ok=True)
        prefix = "Validation/tracking/eval/stereomis"
        render_all = bool(self.config.get("TRACKING_VAL_RENDER_VIDEOS_ALL_SEQS", False))
        rng = random.Random(int(self.config.get("TRACKING_VAL_VIDEO_SEED", 0)) + int(epoch))

        eligible_names: List[str] = []
        for seq_name in val_seqs:
            try:
                ds_try = StereoMISTracking(
                    root=root, sequence=seq_name, height=height, width=width,
                    start=0, stop=stop, step=1,
                )
            except (FileNotFoundError, RuntimeError, OSError):
                continue
            if len(ds_try) >= 2 and ds_try.tracking_points is not None:
                eligible_names.append(seq_name)

        video_seq = rng.choice(eligible_names) if eligible_names else None

        self.temporal_tracker.eval()
        per_seq_rows: List[List[Any]] = []
        video_payload: Dict[str, Any] = {}
        for seq_name in val_seqs:
            safe = str(seq_name).replace("/", "_").replace(os.sep, "_")
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
                self.logger.warning(f"StereoMIS val skip {seq_name}: {e}", context="TRACKING")
                continue
            if len(ds) < 2:
                continue
            if ds.tracking_points is None:
                continue

            all_frames = torch.stack(
                [ds[t]["image"] for t in range(len(ds))], dim=0,
            ).unsqueeze(0).to(self.device)
            h, w = ds[0]["image"].shape[1:]
            grid_pts = make_grid_points(h, w, grid_h=grid_size, grid_w=grid_size)

            with torch.no_grad():
                grid_out = self.temporal_tracker.track_long_sequence(
                    grid_pts.unsqueeze(0).to(self.device),
                    all_frames,
                    window_size=window_size,
                    **self._track_long_sequence_infer_kwargs(),
                )
                grid_traj = grid_out["tracks"].squeeze(0).permute(1, 0, 2).cpu()
                grid_vis_logits = grid_out.get("visibility", None)
                grid_vis_np: torch.Tensor | None = None
                if grid_vis_logits is not None:
                    grid_vis_np = (
                        torch.sigmoid(grid_vis_logits.squeeze(0)) > 0.5
                    ).cpu()  # [N, T]
                gt_init = ds.tracking_points[:, 0, :].to(self.device)
                gt_out = self.temporal_tracker.track_long_sequence(
                    gt_init.unsqueeze(0),
                    all_frames,
                    window_size=window_size,
                    **self._track_long_sequence_infer_kwargs(),
                )
            pred_tracks = gt_out["tracks"].squeeze(0)
            vis_logits = gt_out["visibility"].squeeze(0)
            pred_vis = torch.sigmoid(vis_logits) > 0.5
            T_pred = pred_tracks.shape[1]
            gt_tracks = ds.tracking_points[:, :T_pred, :].to(pred_tracks.device)
            gt_vis = ds.visibility[:, :T_pred].to(pred_tracks.device)
            results = compute_tap_metrics(pred_tracks, gt_tracks, gt_vis, pred_vis)

            per_seq_rows.append([
                seq_name,
                int(gt_tracks.shape[0]),
                int(T_pred),
                results["delta_avg"],
                results["OA"],
                results["AJ"],
            ])
            self.logger.info(
                f"  StereoMIS {seq_name}: δavg={results['delta_avg']:.4f} "
                f"OA={results['OA']:.4f} AJ={results['AJ']:.4f}",
                context="TRACKING",
            )

            do_video = render_all or (seq_name == video_seq)
            if do_video:
                grid_path = os.path.join(out_dir, f"stereomis_{safe}_grid_ep{epoch:04d}.mp4")
                cmp_path = os.path.join(out_dir, f"stereomis_{safe}_cmp_ep{epoch:04d}.mp4")
                try:
                    render_tracks_video(
                        dataset=ds,
                        trajectories=grid_traj,
                        output_path=grid_path,
                        fps=max(fps, 4),
                        trail_length=max(5, fps),
                        point_radius=3,
                        visibility=grid_vis_np,
                    )
                    pixel_errors = (pred_tracks.cpu() - gt_tracks.cpu()).norm(dim=-1)
                    gt_vid = gt_tracks.permute(1, 0, 2).cpu()
                    pred_vid = pred_tracks.permute(1, 0, 2).cpu()
                    # We now always pass the predicted visibility so the
                    # renderer can draw occluded predictions as hollow rings
                    # plus the outer green/red/orange confusion ring.
                    # ``TRACKING_VAL_COMP_GATE_PREDVIS`` toggles the legacy
                    # behaviour of actually *hiding* predicted-occluded points.
                    gate_pred = bool(self.config.get(
                        "TRACKING_VAL_COMP_GATE_PREDVIS", False,
                    ))
                    render_comparison_video(
                        dataset=ds,
                        pred_trajectories=pred_vid,
                        gt_trajectories=gt_vid,
                        output_path=cmp_path,
                        fps=max(fps, 4),
                        trail_length=max(5, fps),
                        point_radius=3,
                        visibility=gt_vis.cpu(),
                        pred_visibility=pred_vis.cpu(),
                        errors=pixel_errors,
                        hide_pred_when_occluded=gate_pred,
                    )
                except RuntimeError as e:
                    self.logger.warning(f"StereoMIS val video {seq_name}: {e}", context="TRACKING")
                    grid_path = cmp_path = ""
                caption = f"[ep {epoch}] {seq_name}"
                if grid_path and os.path.isfile(grid_path):
                    video_payload[f"{prefix}/dense_video"] = wandb.Video(
                        grid_path, fps=max(fps, 4), format="mp4", caption=caption,
                    )
                if cmp_path and os.path.isfile(cmp_path):
                    video_payload[f"{prefix}/gt_vs_pred_video"] = wandb.Video(
                        cmp_path, fps=max(fps, 4), format="mp4", caption=caption,
                    )

            del all_frames, grid_out, gt_out, ds
            torch.cuda.empty_cache()

        if not per_seq_rows:
            return

        delta_vals = [r[3] for r in per_seq_rows]
        oa_vals = [r[4] for r in per_seq_rows]
        aj_vals = [r[5] for r in per_seq_rows]
        agg_payload: Dict[str, Any] = {
            f"{prefix}/delta_avg_mean": float(np.nanmean(delta_vals)),
            f"{prefix}/OA_mean": float(np.nanmean(oa_vals)),
            f"{prefix}/AJ_mean": float(np.nanmean(aj_vals)),
            f"{prefix}/num_sequences": int(len(per_seq_rows)),
            **self._wandb_epoch_axis_dict(epoch),
        }
        self.logger.info(
            f">> StereoMIS val ({len(per_seq_rows)} seqs): "
            f"δavg={agg_payload[f'{prefix}/delta_avg_mean']:.4f} "
            f"OA={agg_payload[f'{prefix}/OA_mean']:.4f} "
            f"AJ={agg_payload[f'{prefix}/AJ_mean']:.4f}",
            context="TRACKING",
        )
        if self.wandb is not None:
            try:
                table = wandb.Table(
                    columns=["sequence", "num_points", "num_frames", "delta_avg", "OA", "AJ"],
                    data=per_seq_rows,
                )
                payload = {
                    **agg_payload,
                    **video_payload,
                    f"{prefix}/per_sequence": table,
                }
                self.wandb.log(payload)
            except Exception as e:
                self.logger.warning(f"StereoMIS val wandb log: {e}", context="TRACKING")

    def _stir_validation_epoch(self, epoch: int) -> None:
        """STIR val: endpoint accuracy on IR-tattoo GT + one tracked video.

        Metrics are computed against the **IR-derived** ground truth provided by
        the STIR Challenge: ``icgstartseg.png`` / ``icgendseg.png`` give two
        sparse point sets (tattoo centers at the first and last frame of each
        clip). We track the start centers through the clip with the temporal
        tracker and score the predicted endpoint against the nearest GT end
        center at native-resolution STIR thresholds ``(4, 8, 16, 32, 64)`` px.

        All W&B entries use **stable keys** (no per-sequence namespace) so each
        epoch logs onto the **same panel** at ``step/epoch`` rather than
        creating a new panel per randomly-chosen sequence. A per-sequence
        breakdown is logged once per epoch as a ``wandb.Table``.
        """
        if not bool(self.config.get("TRACKING_STIR_VAL", True)):
            return
        import random
        from dataset.loader_phase import merge_phase_video_list
        from dataset.utils import select_videos_by_patterns
        from gatetracker.data import STIRTracking
        from gatetracker.tracking.metrics import (
            STIR_THRESHOLDS_PX,
            compute_stir_endpoint_metrics,
        )
        from gatetracker.utils.visualization import (
            render_comparison_video,
            render_tracks_video,
        )

        datasets_config = self.config.get("DATASETS", {}) or {}
        stir_cfg = datasets_config.get("STIR", None)
        if stir_cfg is None:
            return

        stir_path = stir_cfg.get("PATH", "STIR")
        root = resolve_dataset_filesystem_path(stir_path, "STIR")
        if root is None:
            root = os.path.expandvars(os.path.expanduser(str(stir_path)))
        col_globs = stir_cfg.get("COLLECTION_GLOBS", ["*"])
        camera = str(stir_cfg.get("CAMERA", "left")).lower()
        all_stir = STIRTracking.available_sequences(root, col_globs, camera)
        val_patterns = merge_phase_video_list(stir_cfg, "tracking", "VAL_VIDEOS")
        if val_patterns is None or (isinstance(val_patterns, list) and len(val_patterns) == 0):
            val_seqs = all_stir
        else:
            val_seqs = select_videos_by_patterns(all_stir, val_patterns)
        if not val_seqs:
            return

        few_use, few_frames, few_max_seq = self._tracking_val_fewframes_caps()
        stir_stop: Optional[int] = few_frames if few_use else None
        if few_use:
            val_seqs = val_seqs[:few_max_seq]
            self.logger.info(
                f"STIR val: FEWFRAMES cap → {len(val_seqs)} seq(s), stop={stir_stop}",
                context="TRACKING",
            )

        height = int(self.height)
        width = int(self.width)
        window_size = int(
            self.config.get(
                "TRACKING_SEQUENCE_LENGTH",
                self.config.get("TRACKING_WINDOW_SIZE", 16),
            )
        )
        fps = int(stir_cfg.get("FPS", 8))
        out_dir = os.path.join(self.RUN_DIR, "tracking_val")
        os.makedirs(out_dir, exist_ok=True)
        prefix = "Validation/tracking/eval/stir"
        rng = random.Random(int(self.config.get("TRACKING_VAL_VIDEO_SEED", 0)) + int(epoch) + 7919)

        # Pre-filter to sequences that will actually survive the metric loop
        # below (loadable, ``len(ds) >= 2``, non-empty start/end segmentations)
        # so the single random video pick never lands on a skipped sequence.
        # Mirrors the pattern used in ``_stereomis_validation_epoch``.
        eligible_names: List[str] = []
        for seq_name in val_seqs:
            try:
                ds_try = STIRTracking(
                    root=root, sequence=seq_name, height=height, width=width,
                    start=0, stop=stir_stop, step=1,
                )
            except (FileNotFoundError, RuntimeError, OSError):
                continue
            if (
                len(ds_try) >= 2
                and ds_try.start_points.numel() > 0
                and ds_try.end_points_orig.numel() > 0
            ):
                eligible_names.append(seq_name)
        video_seq = rng.choice(eligible_names) if eligible_names else None

        self.temporal_tracker.eval()
        per_seq_rows: List[List[Any]] = []
        video_payload: Dict[str, Any] = {}
        for seq_name in val_seqs:
            safe = str(seq_name).replace("/", "_").replace(os.sep, "_")
            try:
                ds = STIRTracking(
                    root=root, sequence=seq_name, height=height, width=width,
                    start=0, stop=stir_stop, step=1,
                )
            except (FileNotFoundError, RuntimeError, OSError) as e:
                self.logger.warning(f"STIR val skip {seq_name}: {e}", context="TRACKING")
                continue
            if len(ds) < 2:
                continue

            start_pts = ds.start_points  # [N_start, 2] in processing pixels
            end_pts_orig = ds.end_points_orig  # [N_end, 2] in native pixels
            h_orig, w_orig = ds.orig_size
            if start_pts.numel() == 0 or end_pts_orig.numel() == 0:
                self.logger.warning(
                    f"STIR val {seq_name}: empty start/end segmentations", context="TRACKING",
                )
                continue

            all_frames = torch.stack(
                [ds[t]["image"] for t in range(len(ds))], dim=0,
            ).unsqueeze(0).to(self.device)  # [1, T, 3, H, W]
            T_total = int(all_frames.shape[1])

            with torch.no_grad():
                gt_out = self.temporal_tracker.track_long_sequence(
                    start_pts.unsqueeze(0).to(self.device),
                    all_frames,
                    window_size=window_size,
                    **self._track_long_sequence_infer_kwargs(),
                )
            pred_tracks = gt_out["tracks"].squeeze(0)  # [N_start, T, 2] in processing px
            pred_vis_logits_stir = gt_out.get("visibility", None)
            pred_vis_stir: torch.Tensor | None = None
            if pred_vis_logits_stir is not None:
                pred_vis_stir = (
                    torch.sigmoid(pred_vis_logits_stir.squeeze(0)) > 0.5
                ).cpu()  # [N_start, T]

            # Scale predicted endpoints back to native STIR resolution so the
            # [4, 8, 16, 32, 64] px thresholds match the STIRMetrics protocol.
            sx = float(w_orig) / max(float(width), 1.0)
            sy = float(h_orig) / max(float(height), 1.0)
            pred_endpoint_orig = pred_tracks[:, -1, :].clone()  # [N_start, 2]
            pred_endpoint_orig[..., 0] *= sx
            pred_endpoint_orig[..., 1] *= sy
            metric = compute_stir_endpoint_metrics(
                pred_endpoint_orig.cpu(), end_pts_orig, thresholds=STIR_THRESHOLDS_PX,
            )

            per_seq_rows.append([
                seq_name,
                int(metric["num_query_points"]),
                int(metric["num_gt_points"]),
                metric["delta_avg"],
                metric["mean_dist_px"],
                metric["median_dist_px"],
                *[metric[f"acc_{int(th)}px"] for th in STIR_THRESHOLDS_PX],
            ])
            self.logger.info(
                f"  STIR {seq_name}: δavg={metric['delta_avg']:.4f} "
                f"mean={metric['mean_dist_px']:.2f}px N={int(metric['num_query_points'])}",
                context="TRACKING",
            )

            if seq_name == video_seq:
                # Build a **sparse** per-query GT trajectory: start pos at
                # t=0, matched end-center at t=T-1, linearly interpolated in
                # between. Visibility is True only at the two keyframes so the
                # GT overlay hides the unknown intermediate frames.
                end_pts_proc = ds.end_points.to(pred_tracks.device)  # [N_end, 2]
                if end_pts_proc.numel() > 0:
                    pred_end_proc = pred_tracks[:, -1, :]  # [N_start, 2] in proc
                    dmat = torch.cdist(
                        pred_end_proc.unsqueeze(0), end_pts_proc.unsqueeze(0),
                    ).squeeze(0)  # [N_start, N_end]
                    nn_end_idx = dmat.argmin(dim=1)  # [N_start]
                    matched_end = end_pts_proc[nn_end_idx]  # [N_start, 2]
                else:
                    matched_end = pred_tracks[:, -1, :].clone()

                N_start = pred_tracks.shape[0]
                alpha = torch.linspace(
                    0.0, 1.0, T_total, device=pred_tracks.device,
                ).view(1, T_total, 1)  # [1, T, 1]
                gt_traj_proc = (1.0 - alpha) * start_pts.to(
                    pred_tracks.device
                ).unsqueeze(1) + alpha * matched_end.unsqueeze(1)
                # [N_start, T, 2]
                gt_vis = torch.zeros(N_start, T_total, dtype=torch.bool)
                gt_vis[:, 0] = True
                gt_vis[:, -1] = True

                pred_errors = (
                    pred_tracks.cpu() - gt_traj_proc.cpu()
                ).norm(dim=-1)  # [N_start, T]
                vid_grid = os.path.join(out_dir, f"stir_{safe}_grid_ep{epoch:04d}.mp4")
                vid_cmp = os.path.join(out_dir, f"stir_{safe}_cmp_ep{epoch:04d}.mp4")
                try:
                    render_tracks_video(
                        dataset=ds,
                        trajectories=pred_tracks.permute(1, 0, 2).cpu(),  # [T, N, 2]
                        output_path=vid_grid,
                        fps=max(fps, 4),
                        trail_length=max(5, fps),
                        point_radius=3,
                        visibility=pred_vis_stir,
                    )
                    render_comparison_video(
                        dataset=ds,
                        pred_trajectories=pred_tracks.permute(1, 0, 2).cpu(),
                        gt_trajectories=gt_traj_proc.permute(1, 0, 2).cpu(),
                        output_path=vid_cmp,
                        fps=max(fps, 4),
                        trail_length=max(5, fps),
                        point_radius=3,
                        visibility=gt_vis,
                        pred_visibility=pred_vis_stir,
                        errors=pred_errors,
                        gate_prediction_on_gt_vis=False,
                    )
                except RuntimeError as e:
                    self.logger.warning(
                        f"STIR val video {seq_name}: {e}", context="TRACKING",
                    )
                    vid_grid = vid_cmp = ""
                caption = f"[ep {epoch}] {seq_name}"
                if vid_grid and os.path.isfile(vid_grid):
                    video_payload[f"{prefix}/dense_video"] = wandb.Video(
                        vid_grid, fps=max(fps, 4), format="mp4", caption=caption,
                    )
                if vid_cmp and os.path.isfile(vid_cmp):
                    video_payload[f"{prefix}/gt_vs_pred_video"] = wandb.Video(
                        vid_cmp, fps=max(fps, 4), format="mp4", caption=caption,
                    )

            del ds, all_frames, gt_out, pred_tracks
            torch.cuda.empty_cache()

        if not per_seq_rows:
            return

        delta_vals = [r[3] for r in per_seq_rows]
        mean_dist_vals = [r[4] for r in per_seq_rows]
        median_dist_vals = [r[5] for r in per_seq_rows]
        threshold_acc = {
            int(th): [r[6 + j] for r in per_seq_rows]
            for j, th in enumerate(STIR_THRESHOLDS_PX)
        }
        agg_payload: Dict[str, Any] = {
            f"{prefix}/delta_avg_mean": float(np.nanmean(delta_vals)),
            f"{prefix}/mean_dist_px_mean": float(np.nanmean(mean_dist_vals)),
            f"{prefix}/median_dist_px_mean": float(np.nanmean(median_dist_vals)),
            f"{prefix}/num_sequences": int(len(per_seq_rows)),
            **{
                f"{prefix}/acc_{th}px_mean": float(np.nanmean(threshold_acc[th]))
                for th in threshold_acc
            },
            **self._wandb_epoch_axis_dict(epoch),
        }
        self.logger.info(
            f">> STIR val ({len(per_seq_rows)} seqs): "
            f"δavg={agg_payload[f'{prefix}/delta_avg_mean']:.4f} "
            f"mean_dist={agg_payload[f'{prefix}/mean_dist_px_mean']:.2f}px",
            context="TRACKING",
        )
        if self.wandb is not None:
            try:
                table = wandb.Table(
                    columns=[
                        "sequence",
                        "num_query",
                        "num_gt",
                        "delta_avg",
                        "mean_dist_px",
                        "median_dist_px",
                        *[f"acc_{int(th)}px" for th in STIR_THRESHOLDS_PX],
                    ],
                    data=per_seq_rows,
                )
                payload = {
                    **agg_payload,
                    **video_payload,
                    f"{prefix}/per_sequence": table,
                }
                self.wandb.log(payload)
            except Exception as e:
                self.logger.warning(f"STIR val wandb: {e}", context="TRACKING")

    def _tracking_validate_pseudo_novelview(self, epoch: int) -> None:
        if not bool(self.config.get("TRACKING_VAL_PSEUDO_NOVELVIEW", True)):
            return
        if not hasattr(self, "temporal_tracker") or self.temporal_tracker is None:
            return
        if not hasattr(self, "geometryPipeline") or self.geometryPipeline is None:
            return
        if is_ddp_enabled(self.config) and not is_main_process():
            return

        from gatetracker.data.pseudo_gt import (
            GridConfig,
            PseudoGTGenerator,
            deformation_config_from_run_config,
            occluder_config_from_run_config,
            trajectory_config_from_run_config,
        )
        from gatetracker.tracking.losses import (
            composite_supervision_mask,
            validity_at_tracks_bqt,
        )
        from gatetracker.utils.pseudo_novelview_render import (
            masked_mean_l2_px,
            write_pseudo_gt_vs_pred_video,
        )

        seq_ds = self._tracking_datasets.get("Validation")
        if seq_ds is None or len(seq_ds) == 0:
            return
        bi = int(self.config.get("TRACKING_VAL_PSEUDO_BATCH_INDEX", 0))
        bi = max(0, min(bi, len(seq_ds) - 1))
        win = seq_ds[bi]
        frames = win["frames"].unsqueeze(0).to(self.device)  # [1, T, 3, H, W]
        T = frames.shape[1]
        H, W = frames.shape[3], frames.shape[4]

        grid_sz = int(self.config.get("TRACKING_VAL_PSEUDO_GRID_SIZE", 8))
        stride = max(1, int(self.config.get("TRACKING_VAL_PSEUDO_SPARSE_STRIDE", 2)))
        max_pts = int(self.config.get("TRACKING_VAL_PSEUDO_MAX_POINTS", 64))
        seed = 100_003 * int(epoch) + 17

        gen = PseudoGTGenerator(H, W, device=str(self.device))
        traj_cfg = trajectory_config_from_run_config(self.config, n_frames=T)
        deform_cfg = deformation_config_from_run_config(self.config)
        occ_cfg = occluder_config_from_run_config(self.config)
        with torch.no_grad():
            depth, _, K = self.geometryPipeline.compute_geometry(
                frames[:, 0], return_normalized=False,
            )
        res = gen.generate(
            image=frames[:, 0],
            depth=depth,
            intrinsics=K,
            trajectory=traj_cfg,
            deformation=deform_cfg,
            grid=GridConfig(
                grid_size=grid_sz,
                margin_frac=float(self.config.get("PSEUDO_GT_GRID_MARGIN_FRAC", 0.03)),
            ),
            occluders=occ_cfg,
            # seed=seed,
            randomize_trajectory=False,
            frame_valid_erode_px=int(self.config.get("PSEUDO_GT_MASK_ERODE_PX", 0)),
        )
        frames_in = res.frames.unsqueeze(0).to(self.device)  # [1, T, 3, H, W]
        tracks_tq2 = res.tracks  # [T, Q, 2]
        vis_tq = res.visibility.to(device=self.device, dtype=torch.float32)
        qpix = res.query_pixels  # [Q, 2]
        Qfull = tracks_tq2.shape[1]
        idx = torch.arange(0, Qfull, stride, device=self.device, dtype=torch.long)
        if idx.numel() > max_pts:
            g = torch.Generator(device=self.device)
            g.manual_seed(seed + 7)
            perm = torch.randperm(Qfull, generator=g, device=self.device)[:max_pts]
            idx = perm.sort().values
        # index_select requires ``index`` on the same device as the indexed tensor
        idx_q = idx.to(qpix.device)
        qp = qpix.index_select(0, idx_q).to(self.device)
        tracks_gt = tracks_tq2[:, idx_q, :].permute(1, 0, 2).unsqueeze(0)
        vis_gt = vis_tq[:, idx_q].permute(1, 0).unsqueeze(0)
        fv = res.frame_valid.unsqueeze(0).to(self.device)
        w_rgb = validity_at_tracks_bqt(fv, tracks_gt, H, W)
        comp = composite_supervision_mask(vis_gt, w_rgb, tracks_gt, H, W)

        self.temporal_tracker.eval()
        with torch.no_grad():
            out = self.temporal_tracker.track(qp.unsqueeze(0), frames_in)
        pred = out["tracks"]
        pred_vis_logits = out.get("visibility")  # [1, Q, T]
        mean_l2 = masked_mean_l2_px(pred, tracks_gt, comp > 0.5)

        out_dir = os.path.join(self.RUN_DIR, "tracking_val")
        os.makedirs(out_dir, exist_ok=True)
        vid_path = os.path.join(out_dir, f"pseudo_novelview_ep{epoch:04d}.mp4")
        tr_g = tracks_gt.squeeze(0).permute(1, 0, 2).cpu()
        tr_p = pred.squeeze(0).permute(1, 0, 2).cpu()
        vis_draw = vis_gt.squeeze(0).cpu() > 0.5  # [Q, T]
        pred_vis_qt = None
        if pred_vis_logits is not None:
            pred_vis_qt = (torch.sigmoid(pred_vis_logits.squeeze(0)).cpu() > 0.5)
        try:
            write_pseudo_gt_vs_pred_video(
                frames_in.detach().cpu(),
                tr_g,
                tr_p,
                vis_draw,
                vid_path,
                fps=8,
                trail_length=min(8, T),
                predicted_visibility_qt=pred_vis_qt,
            )
        except Exception as e:
            self.logger.warning(f"pseudo novelview video: {e}", context="TRACKING")
            vid_path = ""

        pfx = "Validation/tracking/eval/pseudo_novelview"
        payload: Dict[str, Any] = {
            f"{pfx}/mean_l2_px": float(mean_l2.detach().cpu()),
            **self._wandb_epoch_axis_dict(epoch),
        }
        if self.wandb is not None and vid_path and os.path.isfile(vid_path):
            try:
                payload[f"{pfx}/sparse_gt_vs_pred_video"] = wandb.Video(vid_path, fps=8, format="mp4")
            except Exception as e:
                self.logger.warning(f"pseudo novelview wandb: {e}", context="TRACKING")
        if self.wandb is not None:
            try:
                self.wandb.log(payload)
            except Exception as e:
                self.logger.warning(f"pseudo novelview log: {e}", context="TRACKING")
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
        return unwrap_model(self.matcher.model).save_pretrained_descriptors(path)

    def reinstantiate_model_from_checkpoint(self) -> None:
        """
        Reinstantiate the model from the latest checkpoint saved.
        """
        from gatetracker.distributed_context import unwrap_model

        if normalize_pipeline_phase(self.config) == "tracking":
            self.logger.info(
                "Skipping matcher reinstantiation for PHASE=tracking (descriptor weights "
                "unchanged on disk here; refinement is in "
                f"{self.runname}_tracking_refinement_checkpoint.pth and "
                "tracking_refinement_net.pt).",
                context="GCLOUD",
            )
            return
        self.logger.info(
            f"Attempting reinstatiation from checkpoint @ {self.MODELS_DIR}",
            context="GCLOUD",
        )
        inner = unwrap_model(self.matcher.model)
        inner.fromArtifact(model_name=self.runname, local_path=self.MODELS_DIR)
        self.matcher.model = inner
        self._wrap_matcher_parallel()
        self.logger.info(
            f"Model reinstantiated from checkpoint @ {self.runname}",
            context="GCLOUD",
        )
