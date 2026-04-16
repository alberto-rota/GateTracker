"""
# StereoMIS Tracking Dataset

Dataset class for the StereoMIS Tracking benchmark, designed for evaluating
2D point tracking methods on stereo endoscopic video sequences.

## Data Format

Expected directory structure per sequence:

```
root/
├── P3_1/
│   ├── video_frames/          # Preprocessed stereo frames
│   │   ├── 000001l.png        # Left image
│   │   ├── 000001r.png        # Right image
│   │   └── ...
│   ├── groundtruth.txt        # Camera poses in TUM format (timestamp tx ty tz qx qy qz qw)
│   ├── endoscope_calibration.yaml  # OpenCV stereo calibration
│   ├── track_pts.pckl         # Ground-truth 2D tracking points (pickle)
│   ├── masks/                 # Optional binary tool masks
│   │   ├── 000001l.png
│   │   └── ...
│   └── sequences.txt          # Sub-sequence definitions (optional)
└── P2_1/
    └── ...
```

## References

- StereoMIS Tracking: https://doi.org/10.5281/zenodo.10867949
- robust-pose-estimator: https://github.com/aimi-lab/robust-pose-estimator
- online_endo_track: https://github.com/mhayoz/online_endo_track
"""

import os
import re
import glob
import pickle
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation


# Per-sequence stereo calibration (left camera intrinsics after rectification).
# From online_endo_track configs — used as fallback when YAML calibration is missing.
_INTRINSICS_REGISTRY = {
    "train_P1_1": {"fx": 559.88883, "fy": 559.88883, "cx": 329.73555374, "cy": 257.43183327, "baseline": 2329.1433},
    "train_P2_0": {"fx": 559.88883, "fy": 559.88883, "cx": 329.73555374, "cy": 257.43183327, "baseline": 2329.1433},
    "train_P2_1": {"fx": 559.88883, "fy": 559.88883, "cx": 329.73555374, "cy": 257.43183327, "baseline": 2329.1433},
    "train_P2_2": {"fx": 559.88883, "fy": 559.88883, "cx": 329.73555374, "cy": 257.43183327, "baseline": 2329.1433},
    "train_P2_6": {"fx": 559.88883, "fy": 559.88883, "cx": 329.73555374, "cy": 257.43183327, "baseline": 2329.1433},
    "test_P3_1": {"fx": 559.88883, "fy": 559.88883, "cx": 329.73555374, "cy": 257.43183327, "baseline": 2329.1433},
    "test_P3_2": {"fx": 559.88883, "fy": 559.88883, "cx": 329.73555374, "cy": 257.43183327, "baseline": 2329.1433},
}


def _load_calibration_yaml(path):
    """Load stereo calibration from OpenCV FileStorage YAML.

    Returns left camera intrinsic matrix [3, 3] and stereo baseline scalar.
    """
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Cannot open calibration file: {path}")

    M1 = fs.getNode("M1").mat()  # [3, 3]
    T = fs.getNode("T").mat()    # [3, 1]
    fs.release()

    baseline = float(np.linalg.norm(T))
    K = torch.from_numpy(M1.astype(np.float32))  # [3, 3]
    return K, baseline


def _load_poses_tum(path, scale=1.0):
    """Load poses from TUM/Freiburg format.

    Each line: ``timestamp tx ty tz qx qy qz qw``

    Returns:
        poses: ``[N, 4, 4]`` float32 tensor, normalized to first frame.
        timestamps: ``[N]`` float64 tensor.
    """
    data = np.loadtxt(path)  # [N, 8]
    timestamps = torch.from_numpy(data[:, 0])  # [N]
    translations = data[:, 1:4]  # [N, 3]
    quaternions = data[:, 4:8]   # [N, 4] (qx, qy, qz, qw) — scipy convention

    rotations = Rotation.from_quat(quaternions).as_matrix()  # [N, 3, 3]

    N = len(data)
    poses = np.eye(4, dtype=np.float32)[None].repeat(N, axis=0)  # [N, 4, 4]
    poses[:, :3, :3] = rotations
    poses[:, :3, 3] = translations
    poses = torch.from_numpy(poses)  # [N, 4, 4]

    # Normalize to first frame
    T0_inv = torch.linalg.inv(poses[0:1])  # [1, 4, 4]
    poses = T0_inv @ poses                 # [N, 4, 4]

    # Scale translation
    poses[:, :3, 3] *= scale

    return poses, timestamps


def _load_tracking_points(path):
    """Load ground-truth 2D tracking points from pickle file.

    Returns:
        points:     ``[N_pts, T, 2]`` float32 tensor (pixel coords, -1 if invisible).
        visibility: ``[N_pts, T]``    bool tensor (True = visible).
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    raw_points = data["points"]        # list of N_pts, each list of T entries ([x,y] or None)
    raw_not_visible = data["not_visible"]  # list of N_pts, each list of T bools

    N_pts = len(raw_points)
    T = len(raw_points[0])

    points_np = np.full((N_pts, T, 2), -1.0, dtype=np.float32)
    visibility_np = np.ones((N_pts, T), dtype=bool)

    for i in range(N_pts):
        for t in range(T):
            if raw_not_visible[i][t]:
                visibility_np[i, t] = False
            if raw_points[i][t] is not None:
                points_np[i, t] = raw_points[i][t]
            else:
                visibility_np[i, t] = False

    points = torch.from_numpy(points_np)        # [N_pts, T, 2]
    visibility = torch.from_numpy(visibility_np) # [N_pts, T]
    return points, visibility


class StereoMISTracking(Dataset):
    """
    # StereoMIS Tracking Dataset

    PyTorch Dataset for the StereoMIS Tracking benchmark.  Returns one frame per
    ``__getitem__`` call, suitable for use with a standard ``DataLoader``.
    Sequence-level annotations (tracking points, intrinsics) are exposed as
    properties so they can be accessed outside the data-loading loop.

    ## Constructor

    ```python
    ds = StereoMISTracking(
        root="/path/to/SteroMIS_Tracking",
        sequence="P3_1",
    )
    ```

    ## Per-item output (``__getitem__``)

    | Key            | Shape         | Dtype   | Description                              |
    |----------------|---------------|---------|------------------------------------------|
    | ``index``      | scalar        | int     | Global frame index in the sequence       |
    | ``image``      | ``[3, H, W]``| float32 | Left RGB image, range ``[0, 1]``         |
    | ``image_right``| ``[3, H, W]``| float32 | Right RGB image *(only if load_right)*   |
    | ``pose``       | ``[4, 4]``   | float32 | Camera-to-world pose (first-frame norm.) |
    | ``mask``       | ``[H, W]``   | bool    | Tool mask *(only if load_masks)*         |

    ## Sequence-level properties

    | Property           | Shape               | Description                              |
    |--------------------|---------------------|------------------------------------------|
    | ``tracking_points``| ``[N_pts, T, 2]``   | GT 2D point tracks (px coords, −1=invis) |
    | ``visibility``     | ``[N_pts, T]``      | Per-point per-frame visibility mask       |
    | ``intrinsics``     | ``[3, 3]``          | Left camera intrinsic matrix              |
    | ``baseline``       | scalar              | Stereo baseline (mm)                      |
    | ``num_frames``     | int                 | Total frames in the (sliced) sequence     |
    """

    # Known sequences in the public porcine subset (renamed with train_/test_ prefix)
    SEQUENCES = [
        "train_P1_1", "train_P2_0", "train_P2_1", "train_P2_2", "train_P2_6",
        "test_P3_1", "test_P3_2",
    ]

    def __init__(
        self,
        root: str,
        sequence: str,
        height: int = 512,
        width: int = 640,
        load_right: bool = False,
        load_masks: bool = False,
        start: int = 0,
        stop: int | None = None,
        step: int = 1,
        pose_scale: float = 10.0,
    ):
        """
        Args:
            root:       Path to the StereoMIS Tracking dataset root.
            sequence:   Sequence name (e.g. ``"P3_1"``).
            height:     Target image height (no resizing if matches original).
            width:      Target image width  (no resizing if matches original).
            load_right: Whether to also load the right stereo image.
            load_masks: Whether to load binary tool masks.
            start:      First frame index to include (0-based).
            stop:       Last frame index (exclusive). ``None`` = all frames.
            step:       Frame subsampling step.
            pose_scale: Scaling factor for pose translations (default 10.0,
                        same as online_endo_track).
        """
        super().__init__()

        self.root = root
        self.sequence = sequence
        self.height = height
        self.width = width
        self.load_right = load_right
        self.load_masks = load_masks
        self.pose_scale = pose_scale

        seq_dir = os.path.join(root, sequence)
        if not os.path.isdir(seq_dir):
            raise FileNotFoundError(
                f"Sequence directory not found: {seq_dir}\n"
                f"Available contents in root: {os.listdir(root) if os.path.isdir(root) else 'ROOT NOT FOUND'}"
            )

        # --- Discover left frames ---
        frame_dir = os.path.join(seq_dir, "video_frames")
        if not os.path.isdir(frame_dir):
            raise FileNotFoundError(f"video_frames/ not found in {seq_dir}")

        left_paths = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
        if len(left_paths) == 0:
            raise FileNotFoundError(f"No frames (*.png) found in {frame_dir}")

        # Apply slicing
        left_paths = left_paths[start:stop:step]
        self._left_paths = left_paths
        self._frame_indices = list(range(start, start + len(left_paths) * step, step))

        if load_right:
            self._right_paths = [p.replace("l.png", "r.png") for p in left_paths]

        # --- Masks ---
        if load_masks:
            mask_dir = os.path.join(seq_dir, "masks")
            self._mask_paths = [
                os.path.join(mask_dir, os.path.basename(p))
                for p in left_paths
            ]

        # --- Calibration ---
        # Strip train_/test_ prefix for registry lookup (folders renamed to SCARED convention)
        seq_key = re.sub(r"^(train|test)_", "", sequence)

        # calib_path = os.path.join(seq_dir, "endoscope_calibration.yaml")
        # if os.path.isfile(calib_path):
        #     self._K, self._baseline = _load_calibration_yaml(calib_path)
        self._K, self._baseline = None, None    
        # elif seq_key in _INTRINSICS_REGISTRY:
        #     info = _INTRINSICS_REGISTRY[seq_key]
        #     self._K = torch.tensor([
        #         [info["fx"], 0.0, info["cx"]],
        #         [0.0, info["fy"], info["cy"]],
        #         [0.0, 0.0, 1.0],
        #     ], dtype=torch.float32)  # [3, 3]
        #     self._baseline = info["baseline"]
        # else:
        #     raise FileNotFoundError(
        #         f"No calibration found for sequence {sequence} (key={seq_key}). "
        #         f"Place endoscope_calibration.yaml in {seq_dir}."
        #     )

        # --- Poses ---
        gt_path = os.path.join(seq_dir, "groundtruth.txt")
        if os.path.isfile(gt_path):
            all_poses, all_timestamps = _load_poses_tum(gt_path, scale=pose_scale)
            self._poses = all_poses[start:stop:step]           # [T_slice, 4, 4]
            self._timestamps = all_timestamps[start:stop:step] # [T_slice]
            # Truncate/pad to match frame count
            if len(self._poses) > len(self._left_paths):
                self._poses = self._poses[: len(self._left_paths)]
                self._timestamps = self._timestamps[: len(self._left_paths)]
            elif len(self._poses) < len(self._left_paths):
                pad = len(self._left_paths) - len(self._poses)
                self._poses = torch.cat([
                    self._poses,
                    torch.eye(4, dtype=torch.float32).unsqueeze(0).expand(pad, -1, -1),
                ])
                self._timestamps = torch.cat([
                    self._timestamps,
                    torch.zeros(pad, dtype=torch.float64),
                ])
        else:
            T = len(self._left_paths)
            self._poses = torch.eye(4, dtype=torch.float32).unsqueeze(0).expand(T, -1, -1).clone()
            self._timestamps = torch.zeros(T, dtype=torch.float64)

        # Pre-check if resize is needed and store original dimensions
        self._needs_resize = False
        sample_img = cv2.imread(self._left_paths[0])
        if sample_img is not None:
            self._h_orig, self._w_orig = sample_img.shape[:2]
            if self._h_orig != self.height or self._w_orig != self.width:
                self._needs_resize = True
        else:
            self._h_orig, self._w_orig = self.height, self.width

        # --- Tracking points ---
        track_path = os.path.join(seq_dir, "track_pts.pckl")
        if os.path.isfile(track_path):
            pts, vis = _load_tracking_points(track_path)  # [N_pts, T_full, 2], [N_pts, T_full]
            self._tracking_points = pts[:, start:stop:step, :]  # [N_pts, T_slice, 2]
            self._visibility = vis[:, start:stop:step]           # [N_pts, T_slice]

            # Rescale point coordinates to match the target image dimensions
            if self._needs_resize:
                sx = self.width / self._w_orig    # x scale factor
                sy = self.height / self._h_orig   # y scale factor
                self._tracking_points[:, :, 0] *= sx
                self._tracking_points[:, :, 1] *= sy

                # Mark points that landed outside the target frame as invisible
                out_of_bounds = (
                    (self._tracking_points[:, :, 0] < 0)
                    | (self._tracking_points[:, :, 0] >= self.width)
                    | (self._tracking_points[:, :, 1] < 0)
                    | (self._tracking_points[:, :, 1] >= self.height)
                )
                self._visibility[out_of_bounds] = False

            # Keep only points visible in the first frame
            first_visible = self._visibility[:, 0]  # [N_pts]
            self._tracking_points = self._tracking_points[first_visible]
            self._visibility = self._visibility[first_visible]
        else:
            self._tracking_points = None
            self._visibility = None

    # ------------------------------------------------------------------ #
    #  Properties (sequence-level)
    # ------------------------------------------------------------------ #

    @property
    def tracking_points(self) -> torch.Tensor | None:
        """Ground-truth 2D tracks ``[N_pts, T, 2]`` (float32, pixel coords)."""
        return self._tracking_points

    @property
    def visibility(self) -> torch.Tensor | None:
        """Per-point visibility ``[N_pts, T]`` (bool)."""
        return self._visibility

    @property
    def intrinsics(self) -> torch.Tensor:
        """Left camera intrinsics ``[3, 3]``."""
        return self._K

    @property
    def baseline(self) -> float:
        """Stereo baseline in mm."""
        return self._baseline

    @property
    def poses(self) -> torch.Tensor:
        """All poses ``[T, 4, 4]`` (first-frame normalized)."""
        return self._poses

    @property
    def timestamps(self) -> torch.Tensor:
        """Frame timestamps ``[T]``."""
        return self._timestamps

    @property
    def num_frames(self) -> int:
        return len(self._left_paths)

    # ------------------------------------------------------------------ #
    #  Dataset interface
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self._left_paths)

    def __getitem__(self, idx: int) -> dict:
        out = {}
        out["index"] = self._frame_indices[idx]

        # Left image: BGR -> RGB, [H, W, 3] uint8 -> [3, H, W] float32 [0,1]
        img_l = cv2.imread(self._left_paths[idx])
        img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
        if self._needs_resize:
            img_l = cv2.resize(img_l, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        out["image"] = torch.from_numpy(img_l).permute(2, 0, 1).float() / 255.0  # [3, H, W]

        # Right image (optional)
        if self.load_right:
            img_r = cv2.imread(self._right_paths[idx])
            img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
            if self._needs_resize:
                img_r = cv2.resize(img_r, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            out["image_right"] = torch.from_numpy(img_r).permute(2, 0, 1).float() / 255.0  # [3, H, W]

        # Pose
        out["pose"] = self._poses[idx]  # [4, 4]

        # Mask (optional)
        if self.load_masks:
            mask_path = self._mask_paths[idx]
            if os.path.isfile(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if self._needs_resize:
                    mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
                out["mask"] = torch.from_numpy(mask > 0)  # [H, W] bool
            else:
                out["mask"] = torch.ones(self.height, self.width, dtype=torch.bool)  # [H, W]

        return out

    def __repr__(self) -> str:
        pts_info = f", tracking_pts={self._tracking_points.shape[0]}" if self._tracking_points is not None else ""
        return (
            f"StereoMISTracking(sequence={self.sequence}, frames={len(self)}, "
            f"size=({self.height}, {self.width}){pts_info})"
        )

    @staticmethod
    def available_sequences(root: str) -> list[str]:
        """List sequence directories found under *root*.

        Returns folders that contain a ``video_frames/`` subdirectory.
        Falls back to listing all subdirectories if none match.
        """
        if not os.path.isdir(root):
            return []
        with_frames = sorted([
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
            and os.path.isdir(os.path.join(root, d, "video_frames"))
        ])
        if with_frames:
            return with_frames
        # Fallback: return all subdirectories (ignoring files like scripts)
        return sorted([
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ])
