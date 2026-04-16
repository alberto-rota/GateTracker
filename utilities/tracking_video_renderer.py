import os
import cv2
import numpy as np
import imageio.v2 as imageio


def render_tracking_video(
    dataset,
    mp4_path="tracking_vis.mp4",
    gif_path="tracking_vis.gif",
    max_frames=None,
    max_points=80,
    trail_length=30,
    fps=15,
    point_radius=4,
    trail_thickness=2,
):
    """
    Render trajectory overlay animation from StereoMISTracking dataset.

    Args:
        dataset: StereoMISTracking instance.
        mp4_path: Output MP4 file path.
        gif_path: Output GIF file path.
        max_frames: Optional cap on rendered frames.
        max_points: Number of tracked points to overlay.
        trail_length: Temporal trail length.
        fps: Output frame rate.
        point_radius: Radius of current point marker.
        trail_thickness: Maximum trail thickness.

    Returns:
        (mp4_path_or_none, gif_path_or_none)
    """
    if getattr(dataset, "tracking_points", None) is None:
        print("No tracking points in dataset.")
        return None, None

    pts = dataset.tracking_points.numpy()  # [N_pts, T, 2]
    vis = dataset.visibility.numpy()  # [N_pts, T]
    n_pts, t_total = pts.shape[0], pts.shape[1]
    t_render = min(t_total, len(dataset))
    if max_frames is not None:
        t_render = min(t_render, max_frames)
    n_show = min(n_pts, max_points)

    # Deterministic colors in BGR space
    colors = []
    for i in range(n_show):
        hue = int(180 * i / max(n_show - 1, 1))
        bgr = cv2.cvtColor(
            np.array([[[hue, 255, 220]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
        )[0, 0]
        colors.append(tuple(int(c) for c in bgr))

    # Build RGB frames list for robust GIF output
    frames_rgb = []
    sample0 = dataset[0]
    h, w = sample0["image"].shape[1], sample0["image"].shape[2]  # [3, H, W]

    for t in range(t_render):
        frame_rgb = (
            dataset[t]["image"].permute(1, 2, 0).numpy() * 255.0
        ).astype(np.uint8)  # [H, W, 3]
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        for i in range(n_show):
            if not vis[i, t]:
                continue
            x = int(round(pts[i, t, 0]))
            y = int(round(pts[i, t, 1]))

            # Draw trail
            t_start = max(0, t - trail_length)
            for t_prev in range(t_start, t):
                if not vis[i, t_prev] or not vis[i, t_prev + 1]:
                    continue
                p1 = (int(round(pts[i, t_prev, 0])), int(round(pts[i, t_prev, 1])))
                p2 = (
                    int(round(pts[i, t_prev + 1, 0])),
                    int(round(pts[i, t_prev + 1, 1])),
                )
                alpha = (t_prev - t_start) / max(t - t_start, 1)
                thickness = max(1, int(trail_thickness * alpha))
                cv2.line(frame_bgr, p1, p2, colors[i], thickness, cv2.LINE_AA)

            # Draw current point
            cv2.circle(frame_bgr, (x, y), point_radius, colors[i], -1, cv2.LINE_AA)
            cv2.circle(frame_bgr, (x, y), point_radius, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.putText(
            frame_bgr,
            f"Frame {t}/{t_render}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        frames_rgb.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    gif_ok = False
    mp4_ok = False

    # GIF is usually most portable in notebook + editor preview
    try:
        imageio.mimsave(gif_path, frames_rgb, fps=fps)
        gif_ok = os.path.isfile(gif_path) and os.path.getsize(gif_path) > 0
    except Exception as exc:
        print(f"GIF export failed: {exc}")

    # MP4 may fail depending on OpenCV build/codec support
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(mp4_path, fourcc, fps, (w, h))
        if writer.isOpened():
            for fr_rgb in frames_rgb:
                writer.write(cv2.cvtColor(fr_rgb, cv2.COLOR_RGB2BGR))
            writer.release()
            mp4_ok = os.path.isfile(mp4_path) and os.path.getsize(mp4_path) > 0
        else:
            print("OpenCV MP4 writer unavailable in this environment.")
    except Exception as exc:
        print(f"MP4 export failed: {exc}")

    return (mp4_path if mp4_ok else None), (gif_path if gif_ok else None)
