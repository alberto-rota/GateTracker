"""
Centralized metrics logging with Phase/Category/Metric naming.

All metrics follow the pattern: {Phase}/{Category}/{Metric}
where Phase is Training/Validation/Test, Category groups related metrics,
and Metric is the individual measurement name.
"""

import wandb
from typing import Dict, Optional, Any

from gatetracker.utils.logger import get_logger
from gatetracker.utils.formatting import (
    align,
    metrics_for_wandb,
    abbrev_wandb_run_tag,
    abbrev_console_metric_name,
)

logger = get_logger(__name__)


def register_wandb_step_axes(wb: Any) -> None:
    """Register custom W&B x-axes for train/val/test batch counters and epochs.

    Call once immediately after ``wandb.init`` (same process as training). Uses
    explicit step keys in each ``run.log`` dict so batch vs epoch charts stay
    aligned (see Weights & Biases customize log axes / ``define_metric`` docs).

    Step keys (logged as scalars alongside metrics):
        - ``step/train_batch``: monotonic train batch index
        - ``step/val_batch``: monotonic validation batch index
        - ``step/test_batch``: monotonic test dataloader batch index
        - ``step/epoch``: 0-based epoch index for epoch-level summaries
        - ``step/test_eval``: monotonic index for ``run_tests`` / eval media logs
    """
    for axis in (
        "step/train_batch",
        "step/val_batch",
        "step/test_batch",
        "step/epoch",
        "step/test_eval",
    ):
        wb.define_metric(axis)

    # Epoch aggregates (narrow patterns before broader phase globs)
    for phase in ("Training", "Validation"):
        wb.define_metric(f"{phase}/epoch/*", step_metric="step/epoch")
        wb.define_metric(f"{phase}/epoch/refine/*", step_metric="step/epoch")
    wb.define_metric("Test/idx/*", step_metric="step/epoch")

    arch_sub = ("gate", "raw", "fused", "confidence")
    categories = (
        "loss",
        "matching",
        "refinement",
        "fusion",
        "tracking",
        "optim",
        "misc",
    )
    for phase, sm in (
        ("Training", "step/train_batch"),
        ("Validation", "step/val_batch"),
        ("Test", "step/test_batch"),
    ):
        for cat in categories:
            wb.define_metric(f"{phase}/{cat}/*", step_metric=sm)
        wb.define_metric(f"{phase}/images/*", step_metric=sm)
        for sub in arch_sub:
            wb.define_metric(f"{phase}/{sub}/*", step_metric=sm)
        wb.define_metric(f"{phase}/gate/per_layer/*", step_metric=sm)
        wb.define_metric(f"{phase}/tracking/self_sup/*", step_metric=sm)
        wb.define_metric(f"{phase}/tracking/pseudo_gt/*", step_metric=sm)

    # Legacy / watch: gradient + LR keys not under Phase/Category
    wb.define_metric("Gradients/*", step_metric="step/train_batch")
    wb.define_metric("HyperParameters/*", step_metric="step/train_batch")

    # Issues + legacy Step/* (if still present in payloads)
    wb.define_metric("Step/batch")
    wb.define_metric("Step/valbatch")
    wb.define_metric("Step/lossissues")
    wb.define_metric("Issues/*", step_metric="Step/lossissues")

    # Offline-style test / eval logs (not the per-batch Test dataloader counter).
    # W&B only allows a single trailing glob ``*`` (no ``*/*``); deeper paths like
    # ``Test/tracking/<seq>/delta_avg`` still receive ``step/test_eval`` in the
    # same ``log`` dict but may chart on the default step until flattened.
    wb.define_metric("Test/tracking/*", step_metric="step/test_eval")
    wb.define_metric("Test/tracking/eval/*", step_metric="step/test_eval")
    wb.define_metric("Test/Summary", step_metric="step/test_eval")
    wb.define_metric("Validation/tracking/eval/*", step_metric="step/epoch")


METRIC_CATEGORIES = {
    "TotalLoss": "loss",
    "Loss": "loss",
    "InfoNCE": "loss",
    "Fundamental": "loss",
    "Refinement_loss": "loss",
    "tracking_loss": "loss",

    "PrimaryMetric": "matching",
    "F1": "matching",
    "InlierRatio": "matching",
    "NCM": "matching",
    "AUCPR": "matching",
    "EpipolarError": "matching",
    "MDistMean": "matching",
    "MatchCount": "matching",

    "refine_mean_offset": "refinement",
    "refine_std_offset": "refinement",
    "refine_max_offset": "refinement",
    "refine_active_pct": "refinement",
    "refine_score_mean": "refinement",

    "gate_entropy": "fusion",
    "effective_layers_mean": "fusion",
    "max_gate_weight": "fusion",
    "raw_margin_mean": "fusion",
    "fused_margin_mean": "fusion",

    "cycle_error": "tracking",
    "loss_desc": "tracking",
    "coarse_to_fine_delta": "tracking",
    "confidence_mean": "tracking",
    "visibility_ratio": "tracking",

    "LR": "optim",
    "GradNorm": "optim",
    "WeightNorm": "optim",
}


class MetricsLogger:
    """Centralized metric logger supporting console (Rich) and W&B.

    Usage:
        ml = MetricsLogger(wandb_run=wandb_instance)
        ml.log_batch("Training", epoch=1, batch=5, total_batches=100, metrics=raw_dict)
        ml.log_epoch("Training", epoch=1, metrics=epoch_summary)
        ml.log_images("Training", images_dict)
    """

    def __init__(self, wandb_run=None, run_name: str = "run"):
        self.wandb = wandb_run
        self.run_name = run_name
        self._step_counters = {
            "Training": 0,
            "Validation": 0,
            "Test": 0,
        }

    @staticmethod
    def define_wandb_metrics():
        """Register metric step associations with W&B. Call once at init."""
        register_wandb_step_axes(wandb)

    def _categorize_metric(self, key: str) -> str:
        """Map a raw metric name to its category."""
        for pattern, category in METRIC_CATEGORIES.items():
            if pattern.lower() in key.lower():
                return category
        return "misc"

    def _namespace_metrics(self, phase: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Convert flat metric dict to Phase/Category/Metric namespace."""
        namespaced = {}
        for key, value in metrics.items():
            if value is None:
                continue
            if "/" in key:
                namespaced[key] = value
                continue
            category = self._categorize_metric(key)
            namespaced[f"{phase}/{category}/{key}"] = value
        return namespaced

    def log_batch(
        self,
        phase: str,
        epoch: int,
        batch_idx: int,
        total_batches: int,
        metrics: Dict[str, Any],
        extra_info: Optional[str] = None,
        tracking_metrics: Optional[Dict[str, Any]] = None,
    ):
        """Log metrics for a single batch to console and optionally wandb.

        ``tracking_metrics`` (optional) are logged once under ``{phase}/tracking/*``.
        Pass them here instead of merging into ``metrics`` before calling, to avoid
        duplicate W&B rows from a second ``log_tracking`` call.
        """
        self._step_counters[phase] += 1

        run_tag = abbrev_wandb_run_tag(self.run_name)
        tag_col = len(run_tag) + 1
        epoch_batch = f"E{epoch + 1} B{batch_idx + 1}/{total_batches}"
        prefix = align(f"{run_tag}:", tag_col, "right") + epoch_batch
        if extra_info:
            prefix = f"[purple]{extra_info.strip()}[/purple] " + prefix

        metric_parts: list = []
        loss_val = metrics.get("Loss")
        if loss_val is not None:
            try:
                metric_parts.append(f"[yellow]Loss[/yellow]={float(loss_val):.4f}")
            except (TypeError, ValueError):
                pass

        for key, val in metrics.items():
            if key == "Loss" or "/" in key or val is None:
                continue
            try:
                lab = abbrev_console_metric_name(key)
                metric_parts.append(f"[yellow]{lab}[/yellow]={float(val):.4f}")
            except (TypeError, ValueError):
                continue

        if tracking_metrics:
            for tk, val in tracking_metrics.items():
                if val is None:
                    continue
                try:
                    lab = abbrev_console_metric_name(str(tk))
                    metric_parts.append(f"[yellow]{lab}[/yellow]={float(val):.4f}")
                except (TypeError, ValueError):
                    continue
        else:
            # Temporal / other paths may only pass namespaced ``{phase}/tracking/*`` keys.
            _tprefix = f"{phase}/tracking/"
            for key, val in metrics.items():
                if not isinstance(key, str) or not key.startswith(_tprefix) or val is None:
                    continue
                try:
                    tk = key[len(_tprefix) :]
                    lab = abbrev_console_metric_name(str(tk))
                    metric_parts.append(f"[yellow]{lab}[/yellow]={float(val):.4f}")
                except (TypeError, ValueError):
                    continue

        metrs = " ".join(metric_parts)
        logger.info(prefix + (" " + metrs if metrs else ""), context=phase.upper())

        if self.wandb is not None:
            step_key = {"Training": "step/train_batch", "Validation": "step/val_batch", "Test": "step/test_batch"}
            wandb_dict = self._namespace_metrics(phase, metrics)
            # ``Step/*`` keys are for on-disk CSV alignment only; W&B uses ``step/*`` axes.
            wandb_dict = {k: v for k, v in wandb_dict.items() if not k.startswith("Step/")}
            if tracking_metrics:
                for tk, val in tracking_metrics.items():
                    if val is None:
                        continue
                    wandb_dict[f"{phase}/tracking/{tk}"] = val
            wandb_dict[step_key.get(phase, "step/train_batch")] = self._step_counters[phase]
            self.wandb.log(wandb_dict)

    def log_epoch(self, phase: str, epoch: int, metrics: Dict[str, Any]):
        """Log epoch-level summary metrics."""
        if self.wandb is not None:
            epoch_dict = {f"{phase}/epoch/{k}": v for k, v in metrics.items() if v is not None}
            epoch_dict["step/epoch"] = epoch
            self.wandb.log(epoch_dict)

    def log_images(self, phase: str, images: Dict[str, Any]):
        """Log visualization images to wandb only (no disk save)."""
        if self.wandb is None:
            return
        step_key = {"Training": "step/train_batch", "Validation": "step/val_batch", "Test": "step/test_batch"}
        sk = step_key.get(phase, "step/train_batch")
        wandb_images: Dict[str, Any] = {sk: self._step_counters[phase]}
        for name, img in images.items():
            key = f"{phase}/images/{name}"
            if hasattr(img, "save"):
                wandb_images[key] = wandb.Image(img)
            else:
                wandb_images[key] = img
        self.wandb.log(wandb_images)

