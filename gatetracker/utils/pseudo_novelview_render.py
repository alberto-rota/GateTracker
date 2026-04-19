"""Side-by-side MP4: pseudo-GT tracks vs predicted tracks on novel-view frames."""

import os
import shutil
import tempfile
import numpy as np
import torch


def write_pseudo_gt_vs_pred_video(
    frames_btchw: torch.Tensor,
    tracks_gt_bt: torch.Tensor,
    tracks_pred_bt: torch.Tensor,
    visibility_gt_qt: torch.Tensor,
    output_path: str,
    *,
    fps: int = 8,
    trail_length: int = 8,
    point_radius: int = 2,
) -> None:
    """Write ``[T,3,H,W]`` frames with left=pseudo-GT trails, right=pred trails.

    Args:
        frames_btchw: ``[1, T, 3, H, W]`` float in ``[0, 1]``, device CPU or CUDA.
        tracks_gt_bt: ``[T, Q, 2]`` pseudo-GT pixel coords (same space as frames).
        tracks_pred_bt: ``[T, Q, 2]`` predicted coords.
        visibility_gt_qt: ``[Q, T]`` bool/float — draw GT trail only when visible.
        output_path: Destination ``.mp4``.
        fps: Output FPS.
        trail_length: Polyline history length per point.
        point_radius: CV2 circle radius.
    """
    import cv2

    frames = frames_btchw.detach().float().cpu()[0].permute(0, 2, 3, 1).numpy()  # [T, H, W, 3]
    T, H, W, _ = frames.shape
    tr_g = tracks_gt_bt.detach().float().cpu().numpy().astype(np.int32)  # [T, Q, 2]
    tr_p = tracks_pred_bt.detach().float().cpu().numpy().astype(np.int32)
    vis = visibility_gt_qt.detach().float().cpu().numpy()  # [Q, T]
    if vis.dtype != np.bool_:
        vis = vis > 0.5

    cmap = _hsv_colors(tr_g.shape[1])

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp_path = tempfile.mktemp(suffix=".mp4") if shutil.which("ffmpeg") else output_path
    writer = cv2.VideoWriter(tmp_path, fourcc, max(1, int(fps)), (W * 2, H))

    for t in range(T):
        base = (np.clip(frames[t], 0.0, 1.0) * 255.0).astype(np.uint8)
        base_bgr = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
        left = base_bgr.copy()
        right = base_bgr.copy()
        t0 = max(0, t - int(trail_length))
        for qi in range(tr_g.shape[1]):
            col = tuple(int(c) for c in cmap[qi])
            v_win = vis[qi, t0 : t + 1]
            pts_g = tr_g[t0 : t + 1, qi][v_win]
            if len(pts_g) >= 2:
                cv2.polylines(left, [pts_g], False, col, 1, cv2.LINE_AA)
            elif len(pts_g) == 1:
                cv2.circle(left, tuple(pts_g[0]), int(point_radius), col, -1, cv2.LINE_AA)
            pts_p = tr_p[t0 : t + 1, qi][v_win]
            if len(pts_p) >= 2:
                c2 = tuple(int(x * 0.6) for x in col)
                cv2.polylines(right, [pts_p], False, c2, 1, cv2.LINE_AA)
            elif len(pts_p) == 1:
                c2 = tuple(int(x * 0.6) for x in col)
                cv2.circle(right, tuple(pts_p[0]), int(point_radius), c2, -1, cv2.LINE_AA)
        pair = np.concatenate([left, right], axis=1)
        writer.write(pair)
    writer.release()

    if tmp_path != output_path and shutil.which("ffmpeg"):
        try:
            import subprocess

            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", tmp_path,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-loglevel", "error", output_path,
                ],
                check=False,
            )
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)
        except Exception:
            if os.path.isfile(tmp_path):
                os.replace(tmp_path, output_path)


def _hsv_colors(n: int) -> np.ndarray:
    """BGR uint8 colors ``[n, 3]``."""
    import cv2

    out = np.zeros((max(n, 1), 3), dtype=np.uint8)
    for i in range(n):
        h = int(180 * i / max(n, 1))
        rgb = cv2.cvtColor(np.uint8([[[h, 255, 255]]]), cv2.COLOR_HSV2RGB)[0, 0]
        out[i] = rgb[::-1].astype(np.uint8)
    return out


def masked_mean_l2_px(
    pred_bqt2: torch.Tensor,
    gt_bqt2: torch.Tensor,
    mask_bqt: torch.Tensor,
) -> torch.Tensor:
    """Scalar mean L2 in pixels where ``mask_bqt`` is True. Shapes ``[B,Q,T,2]``, ``[B,Q,T]``."""
    d = (pred_bqt2 - gt_bqt2).norm(dim=-1)  # [B, Q, T]
    m = mask_bqt.to(dtype=d.dtype)
    denom = m.sum().clamp_min(1.0)
    return (d * m).sum() / denom
