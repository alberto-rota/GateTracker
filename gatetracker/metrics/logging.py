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


METRIC_CATEGORIES = {
    "Loss": "loss",
    "InfoNCE": "loss",
    "Fundamental": "loss",
    "Refinement_loss": "loss",
    "tracking_loss": "loss",

    "F1": "matching",
    "InlierRatio": "matching",
    "NCM": "matching",
    "Precision": "matching",
    "Recall": "matching",
    "AUCPR": "matching",
    "EpipolarError": "matching",
    "FundamentalError": "matching",
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
        wandb.define_metric("step/train_batch")
        wandb.define_metric("step/val_batch")
        wandb.define_metric("step/test_batch")
        wandb.define_metric("step/epoch")

        for phase, step in [
            ("Training", "step/train_batch"),
            ("Validation", "step/val_batch"),
            ("Test", "step/test_batch"),
        ]:
            for category in ["loss", "matching", "refinement", "fusion", "tracking", "optim"]:
                wandb.define_metric(f"{phase}/{category}/*", step_metric=step)
            wandb.define_metric(f"{phase}/images/*", step_metric=step)

        wandb.define_metric("Training/epoch/*", step_metric="step/epoch")
        wandb.define_metric("Validation/epoch/*", step_metric="step/epoch")

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
    ):
        """Log metrics for a single batch to console and optionally wandb."""
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

        metrs = " ".join(metric_parts)
        logger.info(prefix + (" " + metrs if metrs else ""), context=phase.upper())

        if self.wandb is not None:
            step_key = {"Training": "step/train_batch", "Validation": "step/val_batch", "Test": "step/test_batch"}
            wandb_dict = self._namespace_metrics(phase, metrics)
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
        wandb_images = {}
        for name, img in images.items():
            key = f"{phase}/images/{name}"
            if hasattr(img, 'save'):
                wandb_images[key] = wandb.Image(img)
            else:
                wandb_images[key] = img
        self.wandb.log(wandb_images)

    def log_tracking(self, phase: str, metrics: Dict[str, float]):
        """Log tracking-specific metrics with dedicated formatting."""
        parts = []
        for key in ["cycle_error", "loss_desc", "coarse_to_fine_delta", "confidence_mean", "visibility_ratio", "loss_total"]:
            val = metrics.get(key)
            if val is not None:
                parts.append(f"{key}={val:.4f}")
        if parts:
            logger.info(f"[Tracking] {' '.join(parts)}", context=phase.upper())

        if self.wandb is not None:
            wandb_dict = {f"{phase}/tracking/{k}": v for k, v in metrics.items() if v is not None}
            self.wandb.log(wandb_dict)
