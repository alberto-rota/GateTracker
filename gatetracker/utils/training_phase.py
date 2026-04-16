"""
Pipeline phase helpers: ``PHASE`` selects pretrain, joint (end2end), or temporal tracking.

``PHASE`` values (case-insensitive):

- ``pretrain``: Descriptor / matching losses only (pairwise tracking disabled).
- ``end2end``: Same loop as pretrain plus pairwise tracking losses when weights are positive.
- ``tracking``: Temporal refinement training (separate loop); loads ``PRETRAINED_DESCRIPTOR_CKPT``.

Legacy: ``TRACKING_MODE: True`` with ``PHASE: pretrain`` still enables pairwise tracking and logs a
deprecation warning — prefer ``PHASE: end2end``.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch.nn as nn

from gatetracker.utils.logger import get_logger

logger = get_logger(__name__)

_VALID_PHASES = frozenset({"pretrain", "end2end", "tracking"})
_WARNED_TRACKING_MODE_LEGACY = False


def normalize_pipeline_phase(config: Any) -> str:
    """Return one of ``pretrain`` | ``end2end`` | ``tracking``."""
    raw = _cfg_get(config, "PHASE", "pretrain")
    p = str(raw).lower().strip()
    if p not in _VALID_PHASES:
        logger.warning(
            f"Unknown PHASE={raw!r}; expected one of {sorted(_VALID_PHASES)}. Using 'pretrain'."
        )
        return "pretrain"
    return p


def pairwise_tracking_enabled(config: Any) -> bool:
    """Whether the pairwise (two-frame) tracking loss runs inside ``run_epoch``."""
    global _WARNED_TRACKING_MODE_LEGACY
    phase = normalize_pipeline_phase(config)
    if phase == "end2end":
        return True
    if phase == "pretrain" and bool(_cfg_get(config, "TRACKING_MODE", False)):
        if not _WARNED_TRACKING_MODE_LEGACY:
            logger.warning(
                "TRACKING_MODE is deprecated: use PHASE: end2end for joint descriptor + pairwise tracking."
            )
            _WARNED_TRACKING_MODE_LEGACY = True
        return True
    return False


def matcher_should_enable_tracking_head(config: Any) -> bool:
    """Instantiate ``MatcherModel.tracking_head`` when pairwise tracking needs it."""
    if pairwise_tracking_enabled(config):
        return True
    return bool(_cfg_get(config, "TRACKING_HEAD", False))


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def build_optimizer_param_groups(model: nn.Module, config: Any) -> List[Dict[str, Any]]:
    """
    Build Adam/SGD param groups with separate LRs for fusion, fine-feature head, tracking head.

    Keys (each defaults to ``LEARNING_RATE`` when omitted):

    - ``LR_FUSION``: ``hierarchical_fusion``
    - ``LR_FINE_FEATURE``: ``fine_feature_head`` (local refinement CNN)
    - ``LR_TRACKING_HEAD``: ``tracking_head`` (pairwise correlation head)

    Any remaining ``requires_grad`` parameters are assigned ``LEARNING_RATE`` under group ``other``.
    """
    lr_default = float(_cfg_get(config, "LEARNING_RATE", 1e-4))
    lr_fusion = float(_cfg_get(config, "LR_FUSION", lr_default))
    lr_fine = float(_cfg_get(config, "LR_FINE_FEATURE", lr_default))
    lr_track = float(_cfg_get(config, "LR_TRACKING_HEAD", lr_default))

    groups: List[Dict[str, Any]] = []
    assigned: set[int] = set()

    def _take(module: nn.Module) -> List[nn.Parameter]:
        out: List[nn.Parameter] = []
        for p in module.parameters():
            if p.requires_grad:
                assigned.add(id(p))
                out.append(p)
        return out

    fusion = getattr(model, "hierarchical_fusion", None)
    if fusion is not None:
        params = _take(fusion)
        if params:
            groups.append(
                {
                    "params": params,
                    "lr": lr_fusion,
                    "group_name": "hierarchical_fusion",
                }
            )

    ff = getattr(model, "fine_feature_head", None)
    if ff is not None:
        params = _take(ff)
        if params:
            groups.append(
                {
                    "params": params,
                    "lr": lr_fine,
                    "group_name": "fine_feature_head",
                }
            )

    th = getattr(model, "tracking_head", None)
    if th is not None:
        params = _take(th)
        if params:
            groups.append(
                {
                    "params": params,
                    "lr": lr_track,
                    "group_name": "tracking_head",
                }
            )

    remaining = [
        p
        for p in model.parameters()
        if p.requires_grad and id(p) not in assigned
    ]
    if remaining:
        groups.append(
            {
                "params": remaining,
                "lr": lr_default,
                "group_name": "other",
            }
        )

    if not groups:
        return [{"params": model.parameters(), "lr": lr_default, "group_name": "all"}]
    return groups
