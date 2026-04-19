"""
STIR (Surgical Tattoos in Infrared) challenge-style sequences for tracking eval.

Expected layout (Zenodo / local extracts)::

    root/
    ├── <collection>/           # e.g. ``03``, ``04``
    │   ├── calib.json
    │   ├── left/
    │   │   ├── seq00/
    │   │   │   └── frames/*.mp4
    │   │   └── ...
    │   └── right/
    │       └── ...
    └── ...
"""

import fnmatch
import glob
import json
import os
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _match_any_glob(name: str, globs: list[str] | None) -> bool:
    if not globs:
        return True
    return any(fnmatch.fnmatch(name, g) for g in globs)


def _read_mp4_rgb_uint8(path: str) -> torch.Tensor:
    """Decode MP4 to ``[T, H, W, C]`` uint8 RGB.

    Prefer ``torchvision.io.read_video`` when PyAV is available; otherwise fall
    back to OpenCV (already required by this repo) so STIR eval does not hard-fail.
    """
    from torchvision.io import read_video

    try:
        vid, _, _ = read_video(path, pts_unit="sec")  # [T0, H0, W0, C] uint8
    except ImportError:
        import cv2

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video (cv2 fallback): {path}") from None
        frames: list[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            raise RuntimeError(f"Empty video read: {path}")
        vid = torch.from_numpy(np.stack(frames, axis=0))
    if vid.numel() == 0:
        raise RuntimeError(f"Empty video read: {path}")
    return vid


class STIRTracking(Dataset):
    """Frame-wise dataset over one STIR visible-light clip (rectified).

    ``sequence`` is a logical id ``"{collection}/{camera}/{seq_dir}"``, e.g.
    ``"03/left/seq00"``, relative to ``root``.
    """

    def __init__(
        self,
        root: str,
        sequence: str,
        height: int = 448,
        width: int = 448,
        start: int = 0,
        stop: Optional[int] = None,
        step: int = 1,
    ):
        super().__init__()
        self.root = root
        self.sequence = sequence.replace("\\", "/").strip("/")
        self.height = int(height)
        self.width = int(width)
        self.start = int(start)
        self.step = int(step) if step else 1

        seq_dir = os.path.join(root, self.sequence, "frames")
        if not os.path.isdir(seq_dir):
            raise FileNotFoundError(f"STIR frames/ not found: {seq_dir}")

        mp4s = sorted(glob.glob(os.path.join(seq_dir, "*.mp4")))
        if not mp4s:
            raise FileNotFoundError(f"No MP4 under {seq_dir}")
        self._video_path = mp4s[0]
        if len(mp4s) > 1:
            # Prefer *visible*.mp4 when multiple (challenge naming).
            vis = [p for p in mp4s if "visible" in os.path.basename(p).lower()]
            if vis:
                self._video_path = vis[0]

        vid = _read_mp4_rgb_uint8(self._video_path)  # [T0, H0, W0, C] uint8
        vid = vid.permute(0, 3, 1, 2).float() / 255.0  # [T0, 3, H0, W0]
        T0 = vid.shape[0]
        s0, s1 = self.start, T0 if stop is None else min(int(stop), T0)
        idx = torch.arange(s0, s1, self.step, dtype=torch.long)
        self._frames = vid.index_select(0, idx).contiguous()  # [T, 3, H0, W0]
        self._h0, self._w0 = int(self._frames.shape[2]), int(self._frames.shape[3])
        self._needs_resize = (self._h0 != self.height) or (self._w0 != self.width)

        parts = self.sequence.split("/")
        self._collection = parts[0] if parts else ""
        self._K = None  # Optional[torch.Tensor]
        self._baseline = None  # Optional[float]
        calib_path = os.path.join(root, self._collection, "calib.json")
        if os.path.isfile(calib_path):
            self._K, self._baseline = _try_parse_stir_calib(calib_path)

        self._tracking_points: torch.Tensor | None = None
        self._visibility: torch.Tensor | None = None
        self._poses = torch.eye(4, dtype=torch.float32).unsqueeze(0).expand(len(self), -1, -1).clone()

    @property
    def tracking_points(self):
        return self._tracking_points

    @property
    def visibility(self):
        return self._visibility

    @property
    def intrinsics(self):
        return self._K

    @property
    def baseline(self):
        return self._baseline

    def __len__(self) -> int:
        return int(self._frames.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        fr = self._frames[idx : idx + 1]  # [1, 3, H0, W0]
        if self._needs_resize:
            fr = F.interpolate(
                fr,
                size=(self.height, self.width),
                mode="bicubic",
                align_corners=False,
            )
        img = fr.squeeze(0)  # [3, H, W]
        return {
            "index": int(idx),
            "image": img,
            "pose": self._poses[idx],
        }

    @staticmethod
    def available_sequences(
        root: str,
        collection_globs=None,
        camera: str = "left",
    ) -> list[str]:
        """List logical sequence ids under *root*."""
        if not os.path.isdir(root):
            return []
        cam = camera.lower()
        if cam not in ("left", "right"):
            raise ValueError(f"camera must be 'left' or 'right', got {camera!r}")
        out: list[str] = []
        for coll in sorted(os.listdir(root)):
            coll_path = os.path.join(root, coll)
            if not os.path.isdir(coll_path) or not _match_any_glob(coll, collection_globs):
                continue
            cam_root = os.path.join(coll_path, cam)
            if not os.path.isdir(cam_root):
                continue
            for seq_name in sorted(os.listdir(cam_root)):
                seq_path = os.path.join(cam_root, seq_name)
                if not os.path.isdir(seq_path):
                    continue
                frames_dir = os.path.join(seq_path, "frames")
                if not os.path.isdir(frames_dir):
                    continue
                if not glob.glob(os.path.join(frames_dir, "*.mp4")):
                    continue
                out.append(f"{coll}/{cam}/{seq_name}")
        return out


def _try_parse_stir_calib(path):
    """Best-effort intrinsics from ``calib.json`` (formats vary)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None, None
    # Common patterns: {"K": [[..],[..],[..]]} or nested left/right
    K_np = None
    if isinstance(data, dict):
        if "K" in data:
            K_np = data["K"]
        elif "left" in data and isinstance(data["left"], dict) and "K" in data["left"]:
            K_np = data["left"]["K"]
    if K_np is None:
        return None, None
    try:
        K = torch.tensor(K_np, dtype=torch.float32).view(3, 3)
    except (TypeError, ValueError):
        return None, None
    return K, None
