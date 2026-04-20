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

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _match_any_glob(name: str, globs: list[str] | None) -> bool:
    if not globs:
        return True
    return any(fnmatch.fnmatch(name, g) for g in globs)


def _contour_centers_from_seg(seg_path: str) -> np.ndarray:
    """Extract bounding-box centers of connected components from a STIR segmentation PNG.

    Mirrors :func:`STIRLoader.getcentersfromseg` so the ground-truth IR tattoo
    locations are in native (1280, 1024) pixel coordinates.

    Returns:
        centers: ``[N, 2]`` float32 array of ``(x, y)`` pixel centers in the
        coordinate system of the segmentation image. Empty ``[0, 2]`` when no
        contours are found.
    """
    if not os.path.isfile(seg_path):
        return np.zeros((0, 2), dtype=np.float32)
    seg = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
    if seg is None:
        return np.zeros((0, 2), dtype=np.float32)
    if seg.ndim == 3:
        seg = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
    seg = (seg > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((0, 2), dtype=np.float32)
    centers = np.empty((len(contours), 2), dtype=np.float32)  # [N, 2]
    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        centers[i, 0] = x + w / 2.0
        centers[i, 1] = y + h / 2.0
    return centers


def _read_mp4_rgb_uint8(
    path: str,
    start: int = 0,
    stop: Optional[int] = None,
    step: int = 1,
) -> torch.Tensor:
    """Decode selected MP4 frames to ``[T, H, W, C]`` uint8 RGB.

    Frames are sampled online (``start:stop:step``) during decode so we do not
    materialize long full clips in memory before slicing.
    """
    start = max(int(start), 0)
    step = max(int(step), 1)
    stop_i = None if stop is None else max(int(stop), 0)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    frames: list[np.ndarray] = []
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx < start:
                frame_idx += 1
                continue
            if stop_i is not None and frame_idx >= stop_i:
                break
            if ((frame_idx - start) % step) == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_idx += 1
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(
            f"Empty video read after slice start={start} stop={stop} step={step}: {path}"
        )
    return torch.from_numpy(np.stack(frames, axis=0))


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

        vid = _read_mp4_rgb_uint8(
            self._video_path, start=self.start, stop=stop, step=self.step,
        )  # [T, H0, W0, C] uint8
        self._frames = vid.permute(0, 3, 1, 2).float().contiguous() / 255.0  # [T, 3, H0, W0]
        self._h0, self._w0 = int(self._frames.shape[2]), int(self._frames.shape[3])
        self._needs_resize = (self._h0 != self.height) or (self._w0 != self.width)

        parts = self.sequence.split("/")
        self._collection = parts[0] if parts else ""
        self._K = None  # Optional[torch.Tensor]
        self._baseline = None  # Optional[float]
        calib_path = os.path.join(root, self._collection, "calib.json")
        if os.path.isfile(calib_path):
            self._K, self._baseline = _try_parse_stir_calib(calib_path)

        # ---- Sparse GT from IR segmentations (start/end tattoo centers) ----
        # STIR only provides the segmentation of the IR-visible tattoos at the
        # first and last frame of the clip. Centers are extracted in the
        # *original* pixel grid of the segmentation PNG (1280x1024 on the
        # public datasets) and mirrored to the (possibly resized) processing
        # grid used by the rest of the pipeline via a simple affine scale.
        seq_dir_full = os.path.join(root, self.sequence)
        start_seg = os.path.join(seq_dir_full, "segmentation", "icgstartseg.png")
        end_seg = os.path.join(seq_dir_full, "segmentation", "icgendseg.png")
        start_np = _contour_centers_from_seg(start_seg)  # [N_start, 2] in orig
        end_np = _contour_centers_from_seg(end_seg)      # [N_end,   2] in orig
        sx = self.width / max(self._w0, 1)
        sy = self.height / max(self._h0, 1)
        self._start_points_orig = torch.from_numpy(start_np).float()  # [N_start, 2]
        self._end_points_orig = torch.from_numpy(end_np).float()      # [N_end,   2]
        scale_xy = torch.tensor([sx, sy], dtype=torch.float32).view(1, 2)
        self._start_points = self._start_points_orig * scale_xy       # [N_start, 2]
        self._end_points = self._end_points_orig * scale_xy           # [N_end,   2]
        self._orig_size = (int(self._h0), int(self._w0))

        # StereoMIS-style ``tracking_points`` / ``visibility`` are not available
        # for STIR (only two keyframes). Kept ``None`` so callers that guard on
        # ``ds.tracking_points is None`` still short-circuit the dense TAP-Vid
        # path.
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

    @property
    def start_points(self) -> torch.Tensor:
        """Tattoo centers at ``t = 0`` in processing pixel coords. ``[N_start, 2]``."""
        return self._start_points

    @property
    def end_points(self) -> torch.Tensor:
        """Tattoo centers at ``t = T-1`` in processing pixel coords. ``[N_end, 2]``."""
        return self._end_points

    @property
    def start_points_orig(self) -> torch.Tensor:
        """Tattoo centers at ``t = 0`` in native segmentation pixel coords. ``[N_start, 2]``."""
        return self._start_points_orig

    @property
    def end_points_orig(self) -> torch.Tensor:
        """Tattoo centers at ``t = T-1`` in native segmentation pixel coords. ``[N_end, 2]``."""
        return self._end_points_orig

    @property
    def orig_size(self) -> tuple[int, int]:
        """``(H_orig, W_orig)`` of the native STIR frames (before resizing)."""
        return self._orig_size

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
