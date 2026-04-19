"""Helpers for phase-aware dataset inclusion and video-list overrides."""

from typing import Any, Mapping, Optional


def infer_dataset_phase(config: Any, dataset_phase: Optional[str]) -> str:
    """Return ``\"pretrain\"`` or ``\"tracking\"`` for loader split logic."""
    if dataset_phase is not None:
        return str(dataset_phase).lower()
    ph = str(config.get("PHASE", "pretrain")).lower()
    return "pretrain" if ph in ("pretrain", "end2end") else "tracking"


def dataset_active_for_phase(dataset_name: str, dataset_config: Mapping[str, Any], active_phase: str) -> bool:
    phases = dataset_config.get("PHASES")
    if phases is None:
        if str(dataset_name).upper() == "STIR":
            phases = ["tracking"]
        else:
            phases = ["pretrain", "tracking"]
    phases_l = [str(p).lower() for p in phases]
    return str(active_phase).lower() in phases_l


def merge_phase_video_list(
    dataset_config: Mapping[str, Any],
    active_phase: str,
    key: str,
) -> Any:
    """Phase-specific override for TRAIN_VIDEOS / VAL_VIDEOS / TEST_VIDEOS."""
    sub = dataset_config.get("PRETRAIN" if active_phase == "pretrain" else "TRACKING", None)
    if isinstance(sub, dict) and sub.get(key) is not None:
        return sub.get(key)
    return dataset_config.get(key)


def cap_video_list(videos: list[str], max_key: str, dataset_config: Mapping[str, Any]) -> list[str]:
    """Deterministic cap: sorted unique order then first N."""
    raw = dataset_config.get(max_key)
    if raw is None or raw == "":
        return videos
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return videos
    if n <= 0 or len(videos) <= n:
        return videos
    return sorted(videos)[:n]
