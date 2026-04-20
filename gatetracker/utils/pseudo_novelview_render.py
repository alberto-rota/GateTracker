"""Side-by-side MP4: pseudo-GT tracks vs predicted tracks on novel-view frames.

Visibility is rendered explicitly on both panels so the reader can tell at a
glance whether the tracker predicts the correct visibility and whether it
recovers the correct position *through* occluded frames.

Marker convention
-----------------
* **visible** — filled coloured disc with a bright trail.
* **occluded** — hollow coloured ring + faded dashed trail; the position is
  still drawn where the GT (or the model's prediction) places it so the
  viewer can visually assess whether the tracker kept the track stable
  while the point was hidden.
"""

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
    point_radius: int = 3,
    predicted_visibility_qt: torch.Tensor | None = None,
) -> None:
    """Write ``[T,3,H,W]`` frames with left = pseudo-GT, right = prediction.

    Both panels draw visibility as filled-vs-hollow markers so the reader
    can directly compare the GT occlusion signal with whatever the
    tracker predicts.

    Args:
        frames_btchw:             ``[1, T, 3, H, W]`` float in ``[0, 1]``.
        tracks_gt_bt:             ``[T, Q, 2]`` pseudo-GT pixel coords.
        tracks_pred_bt:           ``[T, Q, 2]`` predicted pixel coords.
        visibility_gt_qt:         ``[Q, T]`` bool/float — GT visibility.
        output_path:              Destination ``.mp4``.
        fps:                      Output FPS.
        trail_length:             Polyline history length per point.
        point_radius:             CV2 circle radius for the current marker.
        predicted_visibility_qt:  Optional ``[Q, T]`` bool/float predicted
            visibility. When given, the right panel draws predicted-visible
            points filled and predicted-occluded points hollow; the outer
            ring encodes prediction-vs-GT confusion (green = correct,
            red = false-positive, orange = false-negative).
    """
    import cv2

    frames = frames_btchw.detach().float().cpu()[0].permute(0, 2, 3, 1).numpy()  # [T, H, W, 3]
    T, H, W, _ = frames.shape
    tr_g = tracks_gt_bt.detach().float().cpu().numpy()          # [T, Q, 2]
    tr_p = tracks_pred_bt.detach().float().cpu().numpy()        # [T, Q, 2]
    tr_g_i = np.rint(tr_g).astype(np.int32)
    tr_p_i = np.rint(tr_p).astype(np.int32)

    vis = visibility_gt_qt.detach().cpu()
    if vis.dtype != torch.bool:
        vis = vis > 0.5
    vis = vis.numpy()  # [Q, T]

    pv: np.ndarray | None = None
    if predicted_visibility_qt is not None:
        pvt = predicted_visibility_qt.detach().cpu()
        if pvt.dtype != torch.bool:
            pvt = pvt > 0.5
        pv = pvt.numpy()  # [Q, T]

    Q = tr_g.shape[1]
    cmap = _hsv_colors(Q)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp_path = tempfile.mktemp(suffix=".mp4") if shutil.which("ffmpeg") else output_path
    writer = cv2.VideoWriter(tmp_path, fourcc, max(1, int(fps)), (W * 2, H))

    def _draw_marker(
        img: np.ndarray,
        center: tuple[int, int],
        color: tuple[int, int, int],
        is_visible: bool,
        ring_color: tuple[int, int, int] | None = None,
        r: int = 3,
    ) -> None:
        if is_visible:
            cv2.circle(img, center, r, color, -1, cv2.LINE_AA)
            cv2.circle(img, center, r, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            overlay = img.copy()
            cv2.circle(overlay, center, r, color, -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.35, img, 0.65, 0.0, img)
            cv2.circle(img, center, r + 1, color, 1, cv2.LINE_AA)
        if ring_color is not None:
            cv2.circle(img, center, r + 3, ring_color, 1, cv2.LINE_AA)

    def _draw_trail(
        img: np.ndarray,
        pts: np.ndarray,      # [L, 2] int32
        vis_seq: np.ndarray,  # [L] bool
        color: tuple[int, int, int],
    ) -> None:
        if pts.shape[0] < 2:
            return
        faded = tuple(int(0.45 * c + 0.55 * 90) for c in color)
        for k in range(pts.shape[0] - 1):
            p1, p2 = pts[k], pts[k + 1]
            if vis_seq[k] and vis_seq[k + 1]:
                cv2.line(
                    img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
                    color, 1, cv2.LINE_AA,
                )
            else:
                # Dashed segment for (partially) occluded pair.
                seg = np.linalg.norm(p2 - p1)
                n = max(1, int(np.ceil(seg / 6.0)))
                tv = np.linspace(0.0, 1.0, 2 * n + 1)
                for m in range(n):
                    a = p1 + (p2 - p1) * tv[2 * m]
                    b = p1 + (p2 - p1) * tv[2 * m + 1]
                    cv2.line(
                        img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                        faded, 1, cv2.LINE_AA,
                    )

    for t in range(T):
        base = (np.clip(frames[t], 0.0, 1.0) * 255.0).astype(np.uint8)
        base_bgr = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
        left = base_bgr.copy()
        right = base_bgr.copy()
        t0 = max(0, t - int(trail_length))

        for qi in range(Q):
            col = tuple(int(c) for c in cmap[qi])
            col_pred = tuple(int(c * 0.75) for c in col)

            # --- GT panel (left) --------------------------------------
            vis_win_g = vis[qi, t0 : t + 1]
            pts_g = tr_g_i[t0 : t + 1, qi]
            _draw_trail(left, pts_g, vis_win_g, col)
            _draw_marker(
                left,
                (int(tr_g_i[t, qi, 0]), int(tr_g_i[t, qi, 1])),
                col, bool(vis[qi, t]), ring_color=None, r=point_radius,
            )

            # --- Prediction panel (right) ----------------------------
            if pv is None:
                # No predicted visibility: draw prediction gated on GT
                # visibility to match the legacy behaviour.
                vis_win_p = vis_win_g
                pred_is_vis = bool(vis[qi, t])
                ring = None
            else:
                vis_win_p = pv[qi, t0 : t + 1]
                pred_is_vis = bool(pv[qi, t])
                gt_is_vis = bool(vis[qi, t])
                if pred_is_vis == gt_is_vis:
                    ring = (0, 200, 0)
                elif pred_is_vis and not gt_is_vis:
                    ring = (0, 0, 255)
                else:
                    ring = (0, 140, 255)

            pts_p = tr_p_i[t0 : t + 1, qi]
            _draw_trail(right, pts_p, vis_win_p, col_pred)
            _draw_marker(
                right,
                (int(tr_p_i[t, qi, 0]), int(tr_p_i[t, qi, 1])),
                col_pred, pred_is_vis, ring_color=ring, r=point_radius,
            )

        # Panel labels
        cv2.putText(
            left, "pseudo-GT", (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            right, "prediction", (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )

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
