"""
Temporal window wrapper for Mono3D_Dataset / MultiDataset, designed for
training temporal point tracking models with self-supervised losses.

Each ``__getitem__`` returns a window of consecutive frames from a single video,
suitable for the Phase 2 tracking training pipeline.
"""

import os
import torch
from torch.utils.data import Dataset

from dataset.base import Mono3D_Dataset


def _find_video_boundaries(dataset: Mono3D_Dataset):
    """Identify contiguous video segments inside a flat ``rgbpathlist``.

    Returns:
        List of ``(start_idx, end_idx)`` tuples (end exclusive) where each
        range corresponds to frames from the same parent directory.
    """
    paths = dataset.rgbpathlist
    if not paths:
        return []

    boundaries = []
    seg_start = 0
    prev_dir = os.path.dirname(paths[0])

    for i in range(1, len(paths)):
        cur_dir = os.path.dirname(paths[i])
        if cur_dir != prev_dir:
            boundaries.append((seg_start, i))
            seg_start = i
            prev_dir = cur_dir
    boundaries.append((seg_start, len(paths)))
    return boundaries


def _get_sub_datasets(base_dataset):
    """Return a list of Mono3D_Dataset instances from either a plain dataset
    or a MultiDataset (ConcatDataset wrapper).

    Returns:
        list[Mono3D_Dataset]
    """
    if hasattr(base_dataset, "datasets"):
        return list(base_dataset.datasets)
    return [base_dataset]


class SequenceWindowDataset(Dataset):
    """Wraps a :class:`Mono3D_Dataset` or :class:`MultiDataset` to yield
    temporal windows of ``T`` consecutive frames instead of frame pairs.

    Window **start** indices along each video segment are spaced by ``stride``
    (minimum 1) for both ``train`` and ``eval`` modes, so larger strides reduce
    ``len(dataset)`` and speed up epochs.

    Per-item output:

    | Key     | Shape            | Description                       |
    |---------|------------------|-----------------------------------|
    | frames  | [T, 3, H, W]    | RGB images in the window          |
    | paths   | list[str]        | File paths for the T frames       |

    No ground-truth tracks are required — only frames are returned, which
    is sufficient for self-supervised temporal tracking losses.
    """

    def __init__(
        self,
        base_dataset,
        window_size: int = 8,
        stride: int = 4,
        mode: str = "train",
    ):
        """
        Args:
            base_dataset: A :class:`Mono3D_Dataset` or :class:`MultiDataset`.
            window_size:  Number of consecutive frames per window.
            stride:       Step between consecutive window **start** indices when
                          building the index list (both ``train`` and ``eval``).
                          Larger stride ⇒ fewer dataset indices ⇒ shorter epochs;
                          each train ``__getitem__`` still samples uniformly over
                          that list. Minimum effective stride is ``1``.
            mode:         ``"train"`` (random windows) or ``"eval"`` (deterministic by idx).
        """
        super().__init__()
        assert mode in ("train", "eval"), f"mode must be 'train' or 'eval', got {mode}"

        self.window_size = window_size
        self.stride = stride
        self.mode = mode

        self._sub_datasets = _get_sub_datasets(base_dataset)

        # Each entry: (sub_dataset_index, frame_start, frame_end)
        self._window_starts: list[tuple[int, int, int]] = []
        max_seg_len = 0

        for ds_idx, sub_ds in enumerate(self._sub_datasets):
            if not hasattr(sub_ds, "rgbpathlist") or not sub_ds.rgbpathlist:
                continue
            video_segs = _find_video_boundaries(sub_ds)
            for seg_start, seg_end in video_segs:
                seg_len = seg_end - seg_start
                max_seg_len = max(max_seg_len, seg_len)
                if seg_len < window_size:
                    continue
                st = max(1, int(stride))
                for ws in range(seg_start, seg_end - window_size + 1, st):
                    end = min(ws + window_size, seg_end)
                    self._window_starts.append((ds_idx, ws, end))

        if not self._window_starts:
            raise ValueError(
                f"No valid windows of size {window_size} found. "
                f"Largest contiguous segment has {max_seg_len} frames across "
                f"{len(self._sub_datasets)} sub-dataset(s)."
            )

    def __len__(self) -> int:
        return len(self._window_starts)

    def __getitem__(self, idx: int) -> dict:
        if self.mode == "train":
            rand_idx = torch.randint(len(self._window_starts), (1,)).item()
            ds_idx, start, end = self._window_starts[rand_idx]
        else:
            ds_idx, start, end = self._window_starts[idx]

        sub_ds = self._sub_datasets[ds_idx]
        frames = []
        paths = []
        for t in range(start, end):
            img = sub_ds.load_frame(sub_ds.rgbpathlist[t])  # [3, H_raw, W_raw]
            if hasattr(sub_ds, "resize_transform") and sub_ds.resize_transform is not None:
                img = sub_ds.resize_transform(img)  # [3, H, W]
            frames.append(img)
            paths.append(sub_ds.rgbpathlist[t])

        frames = torch.stack(frames, dim=0)  # [T, 3, H, W]
        return {
            "frames": frames,
            "paths": paths,
        }

    def __repr__(self) -> str:
        ds_names = [getattr(d, "name", "?") for d in self._sub_datasets]
        return (
            f"SequenceWindowDataset(datasets={ds_names}, "
            f"windows={len(self)}, window_size={self.window_size}, "
            f"stride={self.stride}, mode={self.mode})"
        )
