"""
Shared tracking inference and visualization utilities.

Provides reusable functions for point tracking evaluation on StereoMIS
sequences, used by both the Engine test pipeline and the standalone
test_stereomis_p3.py script.
"""

import os

import cv2
import numpy as np
import torch

from dataset.stereomis_tracking import StereoMISTracking
from matching.matching import Matcher


def make_grid_points(
    height: int,
    width: int,
    grid_h: int = 10,
    grid_w: int = 10,
    margin: int = 24,
) -> torch.Tensor:
    """Create a uniform grid of query points.

    Returns:
        points: [N=grid_h*grid_w, 2] float32 (x, y) pixel coordinates.
    """
    xs = torch.linspace(margin, width - 1 - margin, steps=grid_w, dtype=torch.float32)
    ys = torch.linspace(margin, height - 1 - margin, steps=grid_h, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)  # [N, 2]
    return points


def propagate_points_nearest(
    points_prev: torch.Tensor,
    src_matches: torch.Tensor,
    tgt_matches: torch.Tensor,
    max_match_distance_px: float,
) -> torch.Tensor:
    """Propagate tracked points via nearest-neighbour lookup in match set.

    Args:
        points_prev: [N, 2] tracked points in source frame coordinates.
        src_matches: [M, 2] matched source points.
        tgt_matches: [M, 2] matched target points.

    Returns:
        points_next: [N, 2] tracked points in target frame coordinates.
    """
    if src_matches.numel() == 0:
        return points_prev

    pairwise_dist = torch.cdist(
        points_prev.unsqueeze(0), src_matches.unsqueeze(0)
    ).squeeze(0)  # [N, M]
    nn_idx = pairwise_dist.argmin(dim=1)  # [N]
    nn_dist = pairwise_dist.gather(1, nn_idx.view(-1, 1)).squeeze(1)  # [N]

    points_candidate = tgt_matches[nn_idx]  # [N, 2]
    valid = nn_dist <= max_match_distance_px  # [N]
    points_next = torch.where(valid.view(-1, 1), points_candidate, points_prev)  # [N, 2]
    return points_next


def _reencode_mp4_for_browser(path: str) -> None:
    """Replace OpenCV ``mp4v`` output at *path* with H.264 when ffmpeg exists.

    MPEG-4 Part 2 in MP4 is poorly supported by Chromium/WebKit HTML5 preview and
    some logging UIs; matches ``pseudo_gt.render_video`` post-processing.
    """
    import shutil
    import subprocess
    import tempfile

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return
    fd, tmp_out = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                path,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-crf",
                "23",
                tmp_out,
            ],
            check=True,
        )
        os.replace(tmp_out, path)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        try:
            os.unlink(tmp_out)
        except OSError:
            pass


def _error_to_bgr(error_px: float, max_error: float = 16.0) -> tuple[int, int, int]:
    """Map a pixel error to a BGR color on a green -> yellow -> red gradient.

    Uses HSV hue: 60 (green, 0 px error) down to 0 (red, >=max_error).
    """
    ratio = min(error_px / max(max_error, 1e-6), 1.0)
    hue = int(60 * (1.0 - ratio))  # 60=green, 30=yellow, 0=red
    bgr = cv2.cvtColor(
        np.array([[[hue, 255, 230]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
    )[0, 0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def render_tracks_video(
    dataset: StereoMISTracking,
    trajectories: torch.Tensor,
    output_path: str,
    fps: int,
    trail_length: int = 20,
    point_radius: int = 3,
    visibility: torch.Tensor | None = None,
    errors: torch.Tensor | None = None,
    error_max_px: float = 16.0,
    hide_occluded: bool = False,
) -> str:
    """Render an MP4 video with trajectory overlays.

    Visibility — when provided — is rendered explicitly rather than used as
    a hard gate so that tracked points remain visible through short
    occlusions. Occluded markers are drawn as **hollow coloured rings with a
    faded alpha fill** and their trail segments are dashed and desaturated.
    Set ``hide_occluded=True`` to fall back to the legacy skip-when-invisible
    behaviour.

    Args:
        dataset: StereoMISTracking instance providing RGB frames.
        trajectories: [T, N, 2] pixel coordinates per frame per point.
        output_path: Destination MP4 file path.
        fps: Output video frame rate.
        trail_length: Number of past frames to draw as trail.
        point_radius: Radius of current-position circles.
        visibility: Optional [N, T] bool tensor. When provided, occluded
            points are drawn as hollow rings (unless ``hide_occluded=True``).
        errors: Optional [N, T] float tensor of per-point per-frame L2 pixel
            errors. Green (0 px) → red (>=error_max_px) gradient, replacing
            the default per-point-index palette.
        error_max_px: Pixel error that saturates to red. Default 16.0.
        hide_occluded: Legacy switch — when True, occluded points are not
            drawn at all (trails & markers skipped).

    Returns:
        The output_path on success.
    """
    t_total, n_pts, _ = trajectories.shape
    h, w = dataset[0]["image"].shape[1:]  # [H, W]

    vis_np = None
    if visibility is not None:
        vis_np = visibility.bool().cpu().numpy()  # [N, T]

    err_np = None
    if errors is not None:
        err_np = errors.float().cpu().numpy()  # [N, T]

    # Fixed per-point-index colors (used when errors is None)
    index_colors = []
    for i in range(n_pts):
        hue = int(180 * i / max(n_pts - 1, 1))
        bgr = cv2.cvtColor(
            np.array([[[hue, 255, 220]]], dtype=np.uint8),
            cv2.COLOR_HSV2BGR,
        )[0, 0]
        index_colors.append(tuple(int(c) for c in bgr))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for: {output_path}")

    trajectories_np = trajectories.cpu().numpy()  # [T, N, 2]

    def _dashed_line_bgr(
        img: np.ndarray,
        p1: tuple[int, int],
        p2: tuple[int, int],
        color: tuple[int, int, int],
        thickness: int = 1,
    ) -> None:
        """Draw a dashed segment (cv2 has no dashed stroke primitive)."""
        v = np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=np.float32)
        dist = float(np.linalg.norm(v))
        if dist < 1e-3:
            return
        n = max(1, int(np.ceil(dist / 6.0)))
        tv = np.linspace(0.0, 1.0, 2 * n + 1, dtype=np.float32)
        p1f = np.array(p1, dtype=np.float32)
        for m in range(n):
            a = p1f + v * tv[2 * m]
            b = p1f + v * tv[2 * m + 1]
            cv2.line(
                img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                color, thickness, cv2.LINE_AA,
            )

    for t in range(t_total):
        frame_rgb = (
            dataset[t]["image"].permute(1, 2, 0).numpy() * 255.0
        ).astype("uint8")  # [H, W, 3]
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        t_start = max(0, t - trail_length)
        for i in range(n_pts):
            is_visible = vis_np is None or bool(vis_np[i, t])
            if hide_occluded and not is_visible:
                continue

            pt_color = (
                _error_to_bgr(err_np[i, t], error_max_px)
                if err_np is not None
                else index_colors[i]
            )
            pt_color_faded = tuple(int(0.45 * c + 0.55 * 90) for c in pt_color)

            if t > 0:
                traj = trajectories_np[t_start : t + 1, i, :]  # [L, 2]
                p1 = traj[:-1].round().astype("int32")
                p2 = traj[1:].round().astype("int32")
                for k in range(p1.shape[0]):
                    seg_t = t_start + k
                    v1 = vis_np is None or bool(vis_np[i, seg_t])
                    v2 = vis_np is None or bool(vis_np[i, seg_t + 1])
                    if hide_occluded and (not v1 or not v2):
                        continue
                    seg_color_base = (
                        _error_to_bgr(err_np[i, seg_t + 1], error_max_px)
                        if err_np is not None
                        else index_colors[i]
                    )
                    alpha = float(k + 1) / max(p1.shape[0], 1)
                    thickness = max(1, int(2 * alpha))
                    if v1 and v2:
                        cv2.line(
                            frame_bgr,
                            (int(p1[k, 0]), int(p1[k, 1])),
                            (int(p2[k, 0]), int(p2[k, 1])),
                            seg_color_base, thickness, cv2.LINE_AA,
                        )
                    else:
                        faded = tuple(int(0.45 * c + 0.55 * 90) for c in seg_color_base)
                        _dashed_line_bgr(
                            frame_bgr,
                            (int(p1[k, 0]), int(p1[k, 1])),
                            (int(p2[k, 0]), int(p2[k, 1])),
                            faded, thickness=1,
                        )

            x, y = trajectories_np[t, i, :]
            center = (int(round(x)), int(round(y)))
            if is_visible:
                cv2.circle(frame_bgr, center, point_radius, pt_color, -1, cv2.LINE_AA)
                cv2.circle(frame_bgr, center, point_radius, (0, 0, 0), 1, cv2.LINE_AA)
            else:
                overlay = frame_bgr.copy()
                cv2.circle(overlay, center, point_radius, pt_color, -1, cv2.LINE_AA)
                cv2.addWeighted(overlay, 0.35, frame_bgr, 0.65, 0.0, frame_bgr)
                cv2.circle(
                    frame_bgr, center, point_radius + 1, pt_color_faded,
                    1, cv2.LINE_AA,
                )

        cv2.putText(
            frame_bgr,
            f"Frame {t+1}/{t_total}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame_bgr)

    writer.release()
    _reencode_mp4_for_browser(output_path)
    return output_path


def render_comparison_video(
    dataset: StereoMISTracking,
    pred_trajectories: torch.Tensor,
    gt_trajectories: torch.Tensor,
    output_path: str,
    fps: int,
    trail_length: int = 20,
    point_radius: int = 3,
    visibility: torch.Tensor | None = None,
    pred_visibility: torch.Tensor | None = None,
    errors: torch.Tensor | None = None,
    error_max_px: float = 16.0,
    gt_color: tuple[int, int, int] = (255, 255, 255),
    gate_prediction_on_gt_vis: bool = True,
    hide_pred_when_occluded: bool = False,
) -> str:
    """Render a comparison MP4 with GT tracks and error-colored predicted tracks.

    GT points are drawn as small hollow circles; when GT visibility is False
    they switch to a **dashed white trail** and are still rendered at the
    GT position so the reader can verify whether the prediction recovers
    to that location. Predicted points are error-coloured filled circles
    when visible and hollow (alpha-blended) error-coloured rings when
    predicted occluded. An **outer confusion ring** (green / red / orange)
    is drawn around the predicted marker whenever both ``visibility`` and
    ``pred_visibility`` are available.

    Args:
        dataset: StereoMISTracking instance providing RGB frames.
        pred_trajectories: [T, N, 2] predicted pixel coordinates.
        gt_trajectories: [T, N, 2] ground-truth pixel coordinates.
        output_path: Destination MP4 file path.
        fps: Output video frame rate.
        trail_length: Number of past frames to draw as trail.
        point_radius: Radius of current-position circles.
        visibility: Optional [N, T] bool tensor (GT visibility).
        pred_visibility: Optional [N, T] bool tensor of predicted visibility.
        errors: Optional [N, T] float tensor of per-point per-frame L2 pixel
            errors used for predicted track coloring. Computed automatically
            from the two trajectories if not provided.
        error_max_px: Pixel error that saturates to red. Default 16.0.
        gt_color: BGR color for GT tracks. Default white (255, 255, 255).
        gate_prediction_on_gt_vis: When True (default legacy), predicted
            tracks are hidden at frames where GT visibility is False. Set
            to False for sparse-GT datasets (STIR) so the predicted
            trajectory keeps being drawn across unlabelled frames.
        hide_pred_when_occluded: When True, predicted markers are skipped
            entirely on frames where ``pred_visibility`` is False (legacy).
            Default False: predicted-occluded points are drawn as hollow
            alpha-blended rings.

    Returns:
        The output_path on success.
    """
    t_total, n_pts, _ = pred_trajectories.shape
    h, w = dataset[0]["image"].shape[1:]  # [H, W]

    vis_np = None
    if visibility is not None:
        vis_np = visibility.bool().cpu().numpy()  # [N, T]

    pred_vis_np = None
    if pred_visibility is not None:
        pred_vis_np = pred_visibility.bool().cpu().numpy()  # [N, T]

    pred_np = pred_trajectories.cpu().numpy()  # [T, N, 2]
    gt_np = gt_trajectories.cpu().numpy()      # [T, N, 2]

    if errors is not None:
        err_np = errors.float().cpu().numpy()  # [N, T]
    else:
        err_np = np.linalg.norm(pred_np - gt_np, axis=-1).T  # [N, T]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for: {output_path}")

    gt_radius = max(1, point_radius - 1)

    def _dashed_line_bgr(
        img: np.ndarray, p1: tuple[int, int], p2: tuple[int, int],
        color: tuple[int, int, int], thickness: int = 1,
    ) -> None:
        v = np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=np.float32)
        dist = float(np.linalg.norm(v))
        if dist < 1e-3:
            return
        n = max(1, int(np.ceil(dist / 6.0)))
        tv = np.linspace(0.0, 1.0, 2 * n + 1, dtype=np.float32)
        p1f = np.array(p1, dtype=np.float32)
        for m in range(n):
            a = p1f + v * tv[2 * m]
            b = p1f + v * tv[2 * m + 1]
            cv2.line(
                img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                color, thickness, cv2.LINE_AA,
            )

    for t in range(t_total):
        frame_rgb = (
            dataset[t]["image"].permute(1, 2, 0).numpy() * 255.0
        ).astype("uint8")  # [H, W, 3]
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        t_start = max(0, t - trail_length)

        for i in range(n_pts):
            gt_visible_here = vis_np is None or bool(vis_np[i, t])
            # Gate the *entire* point (both GT and prediction) on GT vis
            # only for the legacy dense-GT use case.
            skip_point_entirely = (
                gate_prediction_on_gt_vis and vis_np is not None and not vis_np[i, t]
            )
            if skip_point_entirely:
                continue

            # --- GT trail + point --------------------------------------
            gt_faded = tuple(int(0.45 * c + 0.55 * 90) for c in gt_color)
            if t > 0:
                gt_traj = gt_np[t_start : t + 1, i, :]  # [L, 2]
                gp1 = gt_traj[:-1].round().astype("int32")
                gp2 = gt_traj[1:].round().astype("int32")
                for k in range(gp1.shape[0]):
                    seg_t = t_start + k
                    v1 = vis_np is None or bool(vis_np[i, seg_t])
                    v2 = vis_np is None or bool(vis_np[i, seg_t + 1])
                    alpha = float(k + 1) / max(gp1.shape[0], 1)
                    thickness = max(1, int(2 * alpha))
                    if v1 and v2:
                        cv2.line(
                            frame_bgr,
                            (int(gp1[k, 0]), int(gp1[k, 1])),
                            (int(gp2[k, 0]), int(gp2[k, 1])),
                            gt_color, thickness, cv2.LINE_AA,
                        )
                    else:
                        _dashed_line_bgr(
                            frame_bgr,
                            (int(gp1[k, 0]), int(gp1[k, 1])),
                            (int(gp2[k, 0]), int(gp2[k, 1])),
                            gt_faded, 1,
                        )

            gx, gy = gt_np[t, i, :]
            gt_center = (int(round(gx)), int(round(gy)))
            if gt_visible_here:
                cv2.circle(frame_bgr, gt_center, gt_radius, gt_color, 1, cv2.LINE_AA)
            else:
                cv2.circle(frame_bgr, gt_center, gt_radius + 1, gt_faded, 1, cv2.LINE_AA)

            # --- Predicted trail + point (error-colored, on top) ------
            pred_color = _error_to_bgr(err_np[i, t], error_max_px)

            if t > 0:
                pred_traj = pred_np[t_start : t + 1, i, :]  # [L, 2]
                pp1 = pred_traj[:-1].round().astype("int32")
                pp2 = pred_traj[1:].round().astype("int32")
                for k in range(pp1.shape[0]):
                    seg_t = t_start + k
                    pv1 = pred_vis_np is None or bool(pred_vis_np[i, seg_t])
                    pv2 = pred_vis_np is None or bool(pred_vis_np[i, seg_t + 1])
                    if hide_pred_when_occluded and (not pv1 or not pv2):
                        continue
                    if gate_prediction_on_gt_vis and vis_np is not None and (
                        not vis_np[i, seg_t] or not vis_np[i, seg_t + 1]
                    ):
                        continue
                    seg_color = _error_to_bgr(err_np[i, seg_t + 1], error_max_px)
                    alpha = float(k + 1) / max(pp1.shape[0], 1)
                    thickness = max(1, int(2 * alpha))
                    if pv1 and pv2:
                        cv2.line(
                            frame_bgr,
                            (int(pp1[k, 0]), int(pp1[k, 1])),
                            (int(pp2[k, 0]), int(pp2[k, 1])),
                            seg_color, thickness, cv2.LINE_AA,
                        )
                    else:
                        faded = tuple(int(0.45 * c + 0.55 * 90) for c in seg_color)
                        _dashed_line_bgr(
                            frame_bgr,
                            (int(pp1[k, 0]), int(pp1[k, 1])),
                            (int(pp2[k, 0]), int(pp2[k, 1])),
                            faded, 1,
                        )

            px, py = pred_np[t, i, :]
            pred_center = (int(round(px)), int(round(py)))
            pred_is_visible = pred_vis_np is None or bool(pred_vis_np[i, t])
            if hide_pred_when_occluded and not pred_is_visible:
                pass
            elif pred_is_visible:
                cv2.circle(frame_bgr, pred_center, point_radius, pred_color, -1, cv2.LINE_AA)
                cv2.circle(frame_bgr, pred_center, point_radius, (0, 0, 0), 1, cv2.LINE_AA)
            else:
                overlay = frame_bgr.copy()
                cv2.circle(overlay, pred_center, point_radius, pred_color, -1, cv2.LINE_AA)
                cv2.addWeighted(overlay, 0.35, frame_bgr, 0.65, 0.0, frame_bgr)
                cv2.circle(
                    frame_bgr, pred_center, point_radius + 1, pred_color, 1, cv2.LINE_AA,
                )

            # Outer confusion ring (needs both masks to be meaningful).
            if pred_vis_np is not None and vis_np is not None:
                gt_v = bool(vis_np[i, t])
                if pred_is_visible == gt_v:
                    ring_bgr = (0, 200, 0)
                elif pred_is_visible and not gt_v:
                    ring_bgr = (0, 0, 255)
                else:
                    ring_bgr = (0, 140, 255)
                cv2.circle(
                    frame_bgr, pred_center, point_radius + 3, ring_bgr, 1, cv2.LINE_AA,
                )

        cv2.putText(
            frame_bgr,
            f"Frame {t+1}/{t_total}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame_bgr)

    writer.release()
    _reencode_mp4_for_browser(output_path)
    return output_path


def infer_tracks_windowed(
    dataset: StereoMISTracking,
    matching_pipeline: Matcher,
    initial_points: torch.Tensor,
    window_size: int = 16,
    num_refinement_iters: int = 2,
) -> torch.Tensor:
    """Track points using sliding overlapping windows.

    Args:
        dataset: StereoMISTracking instance.
        matching_pipeline: Matcher with track_points_window support.
        initial_points: [N, 2] query positions at frame 0.
        window_size: Temporal window length.
        num_refinement_iters: Forward-backward sweeps per window.

    Returns:
        trajectories: [T, N, 2].
    """
    device = matching_pipeline.device
    T = len(dataset)
    N = initial_points.shape[0]
    stride = max(1, window_size // 2)

    all_frames = torch.stack(
        [dataset[t]["image"] for t in range(T)],
        dim=0,
    ).unsqueeze(0).to(device)  # [1, T, 3, H, W]

    trajectories = torch.zeros(T, N, 2, device=device)
    weight_map = torch.zeros(T, N, 1, device=device)

    trajectories[0] = initial_points.to(device)
    weight_map[0] = 1.0

    for w_start in range(0, T - 1, stride):
        w_end = min(w_start + window_size, T)
        if w_end - w_start < 2:
            break

        window_frames = all_frames[:, w_start:w_end]  # [1, T_w, 3, H, W]
        if w_start == 0:
            qp = initial_points.unsqueeze(0).to(device)  # [1, N, 2]
        else:
            qp = trajectories[w_start].unsqueeze(0)  # [1, N, 2]

        with torch.no_grad():
            out = matching_pipeline.track_points_window(
                qp,
                window_frames,
                num_refinement_iters=num_refinement_iters,
            )
        window_tracks = out["tracks"].squeeze(0)  # [N, T_w, 2]

        vis = out.get("visibility")
        if vis is not None:
            conf = torch.sigmoid(vis.squeeze(0))  # [N, T_w]
        else:
            conf = torch.ones(N, w_end - w_start, device=device)

        for local_t in range(w_end - w_start):
            global_t = w_start + local_t
            w = conf[:, local_t].unsqueeze(-1)  # [N, 1]
            trajectories[global_t] = (
                trajectories[global_t] * weight_map[global_t]
                + window_tracks[:, local_t, :] * w
            ) / (weight_map[global_t] + w).clamp_min(1e-6)
            weight_map[global_t] += w

    return trajectories.cpu()


def infer_tracks(
    dataset: StereoMISTracking,
    matching_pipeline: Matcher,
    grid_h: int = 10,
    grid_w: int = 10,
    max_match_distance_px: float = 32.0,
    matches_dir: str | None = None,
    matches_lag_dir: str | None = None,
    matches_lag_frames: int = 16,
    matches_topk: int = 80,
    use_track_points: bool = True,
    initial_points: torch.Tensor | None = None,
) -> torch.Tensor:
    """Track points across a video sequence frame-by-frame.

    Args:
        dataset: StereoMISTracking instance.
        matching_pipeline: Matcher instance.
        grid_h: Grid rows (ignored when initial_points is provided).
        grid_w: Grid columns (ignored when initial_points is provided).
        max_match_distance_px: NN gating threshold for match_images fallback.
        matches_dir: Optional directory to save per-pair match images.
        matches_lag_dir: Optional directory for lagged-pair match images.
        matches_lag_frames: Lag distance for lagged pair visualization.
        matches_topk: Top-k matches to draw.
        use_track_points: Use track_points (True) or match_images+NN (False).
        initial_points: Optional [N, 2] query points instead of grid.

    Returns:
        trajectories: [T, N, 2].
    """
    device = matching_pipeline.device
    t_total = len(dataset)
    h, w = dataset[0]["image"].shape[1:]  # [H, W]

    if initial_points is not None:
        points_prev = initial_points.to(device).float()  # [N, 2]
    else:
        points_prev = make_grid_points(h, w, grid_h=grid_h, grid_w=grid_w).to(device)

    trajectories = [points_prev.detach().cpu()]
    prev_frame = dataset[0]["image"].to(device)  # [3, H, W]

    for t in range(1, t_total):
        curr_frame = dataset[t]["image"].to(device)  # [3, H, W]

        if use_track_points:
            with torch.no_grad():
                track_out = matching_pipeline.track_points(
                    points_prev, prev_frame, curr_frame
                )
            points_next = track_out["tracked_points"].squeeze(0)  # [Q, 2]
        else:
            with torch.no_grad():
                match_out = matching_pipeline.match_images(
                    prev_frame, curr_frame, knn=1
                )
                src_matches = match_out["source_pixels_matched"].float()  # [M, 2]
                tgt_matches = match_out["target_pixels_matched"].float()  # [M, 2]
                match_scores = match_out["scores"].float()  # [M]

            if matches_dir is not None:
                from utilities.visualization import viewComparePixelMatches

                match_img = viewComparePixelMatches(
                    img1=prev_frame.detach().cpu(),
                    img2=curr_frame.detach().cpu(),
                    pts1=src_matches.detach().cpu(),
                    pts2=tgt_matches.detach().cpu(),
                    pts2_true=tgt_matches.detach().cpu(),
                    scores=match_scores.detach().cpu(),
                    topk=matches_topk,
                    use_actual_topk=True,
                    non_random_colors=True,
                )
                match_img.save(
                    os.path.join(matches_dir, f"{t - 1:04d}_{t:04d}.png"),
                    format="PNG",
                )

            if matches_lag_dir is not None and t >= matches_lag_frames:
                from utilities.visualization import viewComparePixelMatches

                lag_t = t - matches_lag_frames
                lag_frame = dataset[lag_t]["image"].to(device)
                with torch.no_grad():
                    lag_match_out = matching_pipeline.match_images(
                        lag_frame, curr_frame, knn=1
                    )
                    lag_src_matches = lag_match_out["source_pixels_matched"].float()
                    lag_tgt_matches = lag_match_out["target_pixels_matched"].float()
                    lag_scores = lag_match_out["scores"].float()

                lag_match_img = viewComparePixelMatches(
                    img1=lag_frame.detach().cpu(),
                    img2=curr_frame.detach().cpu(),
                    pts1=lag_src_matches.detach().cpu(),
                    pts2=lag_tgt_matches.detach().cpu(),
                    pts2_true=lag_tgt_matches.detach().cpu(),
                    scores=lag_scores.detach().cpu(),
                    topk=matches_topk,
                    use_actual_topk=True,
                    non_random_colors=True,
                )
                lag_match_img.save(
                    os.path.join(matches_lag_dir, f"{lag_t:04d}_{t:04d}.png"),
                    format="PNG",
                )

            points_next = propagate_points_nearest(
                points_prev=points_prev,
                src_matches=src_matches,
                tgt_matches=tgt_matches,
                max_match_distance_px=max_match_distance_px,
            )

        trajectories.append(points_next.detach().cpu())
        points_prev = points_next
        prev_frame = curr_frame

    trajectories = torch.stack(trajectories, dim=0)  # [T, N, 2]
    return trajectories
