"""
StereoMIS point-tracking inference script.

Tracks a 10x10 grid initialized on the first frame of a StereoMIS_Tracking sequence,
propagates points frame-by-frame using MONO3D matcher correspondences, and exports
an MP4 with trajectory overlays.
"""

import argparse
import os
from pathlib import Path

import torch
import yaml
from dotmap import DotMap

from dataset.stereomis_tracking import StereoMISTracking
from matching.matching import Matcher
from utilities.tracking_evaluation import (
    infer_tracks,
    infer_tracks_windowed,
    make_grid_points,
    render_tracks_video,
)


def load_config(config_path: str) -> DotMap:
    with open(config_path, "r", encoding="utf-8") as f:
        config_yaml = yaml.safe_load(f)
    config_parameters = config_yaml["parameters"]
    config_dict = {
        k: v.get("value") for k, v in config_parameters.items() if v is not None
    }
    return DotMap(config_dict)


def resolve_checkpoint_path(checkpoint_ref: str, runs_dir: str = "runs") -> str | None:
    if checkpoint_ref is None:
        return None
    checkpoint_ref = checkpoint_ref.strip()
    if checkpoint_ref == "":
        return None

    expanded = os.path.expanduser(checkpoint_ref)
    if os.path.isfile(expanded):
        return expanded

    looks_like_run_name = (os.sep not in checkpoint_ref) and (
        os.path.splitext(checkpoint_ref)[1] == ""
    )
    if not looks_like_run_name:
        return None

    candidate_paths = [
        os.path.join(runs_dir, checkpoint_ref, "models", f"{checkpoint_ref}_checkpoint.pth"),
        os.path.join(runs_dir, checkpoint_ref, "checkpoints", "weights_best.pt"),
        os.path.join("checkpoints", f"{checkpoint_ref}_checkpoint.pth"),
        os.path.join("checkpoints", f"{checkpoint_ref}.pt"),
    ]
    for candidate in candidate_paths:
        if os.path.isfile(candidate):
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description="StereoMIS P3 tracking demo (10x10 grid, 10 seconds).")
    parser.add_argument("--config", type=str, default="config_test.yaml")
    parser.add_argument("--root", type=str, default="/home/shared/nearmrs/arota/StereoMIS_Tracking")
    parser.add_argument("--sequence", type=str, default="test_P3_1")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path or run name (e.g., wild-feather-787).")
    parser.add_argument("--duration_sec", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=20, help="Video FPS and frame-count budget.")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--grid_h", type=int, default=10)
    parser.add_argument("--grid_w", type=int, default=10)
    parser.add_argument("--max_match_distance_px", type=float, default=32.0)
    parser.add_argument("--output", type=str, default="tracking_vis_p3.mp4")
    parser.add_argument(
        "--matches_dir",
        type=str,
        default="stereomis_pair_matches",
        help="Directory to save per-pair match visualizations.",
    )
    parser.add_argument(
        "--matches_topk",
        type=int,
        default=80,
        help="Top-k matches shown in each pair visualization.",
    )
    parser.add_argument(
        "--matches_lag_frames",
        type=int,
        default=16,
        help="Also save matches between frame t and frame t-lag.",
    )
    parser.add_argument(
        "--matches_lag_dir",
        type=str,
        default="stereomis_pair_matches_lag16",
        help="Directory to save lagged-pair match visualizations.",
    )
    parser.add_argument(
        "--no_track_points",
        action="store_true",
        help="Fall back to match_images + nearest-neighbour propagation instead of track_points.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate tracking against GT points from track_pts.pckl (requires GT).",
    )
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="Use sliding-window multi-frame tracking (track_points_window) instead of pairwise.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=16,
        help="Temporal window size for --windowed mode.",
    )
    parser.add_argument(
        "--refinement_iters",
        type=int,
        default=2,
        help="Forward-backward refinement iterations for --windowed mode.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config.BATCH_SIZE = 1
    config.IMAGE_HEIGHT = args.height
    config.IMAGE_WIDTH = args.width

    sequence = args.sequence
    if not os.path.isdir(os.path.join(args.root, sequence)):
        available = StereoMISTracking.available_sequences(args.root)
        if len(available) == 0:
            raise FileNotFoundError(
                f"No sequence found under root={args.root}. Please set --root correctly."
            )
        # Prefer a P3 sequence when available, otherwise take the first one.
        p3_candidates = [s for s in available if "P3" in s]
        sequence = p3_candidates[0] if len(p3_candidates) > 0 else available[0]

    max_frames = max(2, int(round(args.duration_sec * args.fps)))
    dataset = StereoMISTracking(
        root=args.root,
        sequence=sequence,
        height=args.height,
        width=args.width,
        start=0,
        stop=max_frames,
        step=1,
    )
    if len(dataset) < 2:
        raise RuntimeError(f"Sequence {sequence} has <2 frames in the selected window.")
    os.makedirs(args.matches_dir, exist_ok=True)
    os.makedirs(args.matches_lag_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matching_pipeline = Matcher(config=config, model=config.get("RUN", ""), device=device)
    matching_pipeline.model.eval()

    checkpoint_ref = args.checkpoint
    if checkpoint_ref is None:
        checkpoint_ref = config.get("FROM_PRETRAINED_CHECKPOINT", "")
    checkpoint_path = resolve_checkpoint_path(checkpoint_ref)
    if checkpoint_path is not None:
        matching_pipeline.model.fromArtifact(pth_namestring=checkpoint_path)
        print(f"Loaded checkpoint from path: {checkpoint_path}")
    elif checkpoint_ref:
        matching_pipeline.model.fromArtifact(model_name=checkpoint_ref)
        print(f"Loaded checkpoint from artifact/model name: {checkpoint_ref}")
    else:
        print("No checkpoint provided; using current model initialization.")

    use_track_points = not args.no_track_points

    # --- Always track the user-requested grid (used for the video) ---
    if args.windowed:
        h, w = dataset[0]["image"].shape[1:]
        grid_pts = make_grid_points(h, w, grid_h=args.grid_h, grid_w=args.grid_w)
        trajectories = infer_tracks_windowed(
            dataset=dataset,
            matching_pipeline=matching_pipeline,
            initial_points=grid_pts,
            window_size=args.window_size,
            num_refinement_iters=args.refinement_iters,
        )
    else:
        trajectories = infer_tracks(
            dataset=dataset,
            matching_pipeline=matching_pipeline,
            grid_h=args.grid_h,
            grid_w=args.grid_w,
            max_match_distance_px=args.max_match_distance_px,
            matches_dir=args.matches_dir if not use_track_points else None,
            matches_lag_dir=args.matches_lag_dir if not use_track_points else None,
            matches_lag_frames=args.matches_lag_frames,
            matches_topk=args.matches_topk,
            use_track_points=use_track_points,
            initial_points=None,
        )  # [T, N, 2]

    # --- Evaluation: separately track GT-initialized points for metrics ---
    if args.evaluate and dataset.tracking_points is not None:
        from matching.tracking_metrics import compute_tracking_metrics

        gt_init_pts = dataset.tracking_points[:, 0, :]  # [N_pts, 2]
        if args.windowed:
            eval_trajectories = infer_tracks_windowed(
                dataset=dataset,
                matching_pipeline=matching_pipeline,
                initial_points=gt_init_pts,
                window_size=args.window_size,
                num_refinement_iters=args.refinement_iters,
            )
        else:
            eval_trajectories = infer_tracks(
                dataset=dataset,
                matching_pipeline=matching_pipeline,
                grid_h=args.grid_h,
                grid_w=args.grid_w,
                max_match_distance_px=args.max_match_distance_px,
                use_track_points=use_track_points,
                initial_points=gt_init_pts,
            )  # [T, N_pts, 2]

        gt_tracks = dataset.tracking_points  # [N_pts, T_full, 2]
        gt_vis = dataset.visibility           # [N_pts, T_full]
        T_pred = eval_trajectories.shape[0]
        gt_tracks = gt_tracks[:, :T_pred, :]  # [N_pts, T_pred, 2]
        gt_vis = gt_vis[:, :T_pred]            # [N_pts, T_pred]

        pred_tracks = eval_trajectories.permute(1, 0, 2)  # [N_pts, T_pred, 2]
        pred_vis = torch.ones_like(gt_vis)

        results = compute_tracking_metrics(
            pred_tracks, gt_tracks, gt_vis, pred_vis
        )
        print("\n=== TAP-Vid Evaluation ===")
        for k, v in results.items():
            print(f"  {k:>12s}: {v:.4f}")
        print()

    output_path = str(Path(args.output))
    render_tracks_video(
        dataset=dataset,
        trajectories=trajectories,
        output_path=output_path,
        fps=args.fps,
        trail_length=max(5, int(args.fps)),
        point_radius=3,
    )
    mode_str = (
        "windowed" if args.windowed
        else ("track_points" if use_track_points else "match_images+NN")
    )
    print(
        f"Saved tracking video: {output_path}\n"
        f"Mode: {mode_str}\n"
        f"Sequence: {sequence} | Frames: {len(dataset)} | "
        f"Points: {trajectories.shape[1]}"
    )


if __name__ == "__main__":
    main()

