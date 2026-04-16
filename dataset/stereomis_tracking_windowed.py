"""
Windowed dataset wrapper for StereoMIS Tracking, designed for training and
evaluating temporal point tracking models.

Each ``__getitem__`` returns a temporal window of consecutive frames together
with query points and their ground-truth trajectories + visibility masks.
"""

import torch
from torch.utils.data import Dataset

from dataset.stereomis_tracking import StereoMISTracking


class StereoMISTrackingWindowed(Dataset):
    """
    Wraps a :class:`StereoMISTracking` instance to yield temporal windows
    suitable for tracking head training / evaluation.

    Training mode:
        Windows are randomly sampled; query points randomly drawn from GT
        points visible at the window start.

    Eval mode:
        Windows tile the sequence without overlap; all GT points used.

    Per-item output (``__getitem__``):

    | Key          | Shape               | Description                          |
    |--------------|---------------------|--------------------------------------|
    | frames       | [T_w, 3, H, W]     | RGB images in the window             |
    | query_pts    | [Q, 2]             | Query positions at window start      |
    | gt_tracks    | [Q, T_w, 2]        | GT positions across the window       |
    | gt_vis       | [Q, T_w]           | GT visibility per point per frame    |
    """

    def __init__(
        self,
        base_dataset: StereoMISTracking,
        window_size: int = 8,
        num_query_points: int = 64,
        mode: str = "train",
    ):
        """
        Args:
            base_dataset:     Underlying :class:`StereoMISTracking` instance.
            window_size:      Number of consecutive frames per window.
            num_query_points: Max query points per sample (train mode subsamples).
            mode:             ``"train"`` (random windows/points) or ``"eval"``
                              (tiled windows, all points).
        """
        super().__init__()
        assert mode in ("train", "eval"), f"mode must be 'train' or 'eval', got {mode}"

        self.base = base_dataset
        self.window_size = min(window_size, len(base_dataset))
        self.num_query_points = num_query_points
        self.mode = mode

        if base_dataset.tracking_points is None:
            raise ValueError(
                "StereoMISTrackingWindowed requires a dataset with GT tracking points."
            )

        self._gt_tracks = base_dataset.tracking_points  # [N_pts, T, 2]
        self._gt_vis = base_dataset.visibility           # [N_pts, T]
        self._T = len(base_dataset)

        if mode == "eval":
            self._window_starts = list(range(0, self._T - self.window_size + 1, self.window_size))
            if len(self._window_starts) == 0:
                self._window_starts = [0]
        else:
            self._window_starts = list(range(0, self._T - self.window_size + 1))

    def __len__(self) -> int:
        if self.mode == "train":
            return len(self._window_starts)
        return len(self._window_starts)

    def __getitem__(self, idx: int) -> dict:
        if self.mode == "train":
            start_idx = self._window_starts[
                torch.randint(len(self._window_starts), (1,)).item()
            ]
        else:
            start_idx = self._window_starts[idx]

        end_idx = start_idx + self.window_size

        frames = torch.stack(
            [self.base[t]["image"] for t in range(start_idx, end_idx)], dim=0
        )  # [T_w, 3, H, W]

        gt_window = self._gt_tracks[:, start_idx:end_idx, :]  # [N_pts, T_w, 2]
        vis_window = self._gt_vis[:, start_idx:end_idx]         # [N_pts, T_w]

        visible_at_start = vis_window[:, 0]  # [N_pts]
        valid_indices = visible_at_start.nonzero(as_tuple=False).squeeze(-1)  # [V]

        if valid_indices.numel() == 0:
            Q = self.num_query_points if self.mode == "train" else gt_window.shape[0]
            return {
                "frames": frames,
                "query_pts": torch.zeros(Q, 2),
                "gt_tracks": torch.full((Q, self.window_size, 2), -1.0),
                "gt_vis": torch.zeros(Q, self.window_size, dtype=torch.bool),
            }

        if self.mode == "train" and valid_indices.numel() > self.num_query_points:
            perm = torch.randperm(valid_indices.numel())[: self.num_query_points]
            valid_indices = valid_indices[perm]

        query_pts = gt_window[valid_indices, 0, :]  # [Q, 2]
        gt_tracks = gt_window[valid_indices]          # [Q, T_w, 2]
        gt_vis = vis_window[valid_indices]             # [Q, T_w]

        return {
            "frames": frames,       # [T_w, 3, H, W]
            "query_pts": query_pts, # [Q, 2]
            "gt_tracks": gt_tracks, # [Q, T_w, 2]
            "gt_vis": gt_vis,       # [Q, T_w]
        }

    def __repr__(self) -> str:
        return (
            f"StereoMISTrackingWindowed(base={self.base.sequence}, "
            f"windows={len(self)}, window_size={self.window_size}, "
            f"mode={self.mode})"
        )
