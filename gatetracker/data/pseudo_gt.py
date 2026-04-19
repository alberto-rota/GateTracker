"""Pseudo ground-truth dense track generation for point tracking.

Generates synthetic video sequences with known dense point correspondences
from a single RGB image by combining estimated geometry, smooth random
camera trajectories, and time-varying 3D scene deformations.

Usage::

    from gatetracker.data.pseudo_gt import (
        PseudoGTGenerator, TrajectoryConfig, DeformationConfig, GridConfig,
    )

    gen = PseudoGTGenerator(448, 448, device="cuda")
    result = gen.generate(
        image=I0, depth=D0, intrinsics=K,
        trajectory=TrajectoryConfig(n_frames=24),
        deformation=DeformationConfig(),
        grid=GridConfig(grid_size=32),
        seed=42,
    )
    PseudoGTGenerator.log_to_rerun(result)
    PseudoGTGenerator.render_video(result, "tracks.mp4")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from gatetracker.geometry.projections import BackProject, Project
from gatetracker.geometry.transforms import euler2mat

__all__ = [
    "ScalarOrFloatRange",
    "ScalarOrIntRange",
    "TrajectoryConfig",
    "DeformationConfig",
    "GridConfig",
    "PseudoGTResult",
    "PseudoGTGenerator",
    "trajectory_config_from_run_config",
    "deformation_config_from_run_config",
]


# ── sampling helpers ────────────────────────────────────────────────────────


def _uniform(g: torch.Generator, lo: float, hi: float, device) -> float:
    """Sample a scalar uniformly in [lo, hi]."""
    return lo + (hi - lo) * torch.rand(1, device=device, generator=g).item()


def _uniform_int(g: torch.Generator, lo: int, hi: int, device) -> int:
    """Sample an integer uniformly in [lo, hi] inclusive."""
    return torch.randint(lo, hi + 1, (1,), device=device, generator=g).item()


ScalarOrFloatRange = Union[float, Tuple[float, float]]
ScalarOrIntRange = Union[int, Tuple[int, int]]


def _sample_f(g: torch.Generator, x: ScalarOrFloatRange, device) -> float:
    """Fixed float or uniform sample from ``[lo, hi]``."""
    if isinstance(x, tuple):
        return _uniform(g, float(x[0]), float(x[1]), device)
    return float(x)


def _sample_i(g: torch.Generator, x: ScalarOrIntRange, device) -> int:
    """Fixed int or uniform integer in ``[lo, hi]`` inclusive."""
    if isinstance(x, tuple):
        return _uniform_int(g, int(x[0]), int(x[1]), device)
    return int(x)


def _gaussian_smooth_1d(
    signal: torch.Tensor, sigma: float, device
) -> torch.Tensor:
    """Smooth a ``[T, C]`` signal along dim-0 with a 1-D Gaussian kernel."""
    T, C = signal.shape
    if T <= 1:
        return signal

    ks = max(int(4 * sigma) | 1, 3)
    if ks % 2 == 0:
        ks += 1
    # ``F.pad(..., mode="reflect")`` requires padding < length on that axis.
    # Here length is ``T``; pad = ks // 2, so enforce ks <= 2 * T - 1 (odd).
    max_ks = max(2 * T - 1, 1)
    if max_ks % 2 == 0:
        max_ks -= 1
    ks = min(ks, max_ks)
    if ks < 3:
        return signal

    ax = torch.arange(ks, device=device, dtype=torch.float32) - ks // 2
    kernel = torch.exp(-0.5 * (ax / max(sigma, 0.1)) ** 2)
    kernel = (kernel / kernel.sum()).view(1, 1, ks)  # [1, 1, ks]

    sig = signal.T.unsqueeze(1)  # [C, 1, T]  (C acts as batch)
    pad = ks // 2
    sig = F.pad(sig, (pad, pad), mode="reflect")
    out = F.conv1d(sig, kernel)  # [C, 1, T]
    return out.squeeze(1).T  # [T, C]


def _catmull_rom_interp(
    waypoints: torch.Tensor,  # [W, C]
    T: int,
    device,
) -> torch.Tensor:
    """Catmull-Rom spline interpolation of ``W`` waypoints to ``T`` samples.

    Returns ``[T, C]``.
    """
    W, C = waypoints.shape
    t_out = torch.linspace(0, W - 1, T, device=device)  # [T]

    # pad waypoints with duplicated endpoints for boundary tangents
    pts = torch.cat(
        [waypoints[:1], waypoints, waypoints[-1:]], dim=0
    )  # [W+2, C]

    # find segment index for each output sample
    seg = torch.clamp(t_out.long(), 0, W - 2)  # [T]  segment 0..W-2
    frac = t_out - seg.float()  # [T]  local param in [0, 1)

    p0 = pts[seg]      # [T, C]   (padded index: seg+0 in original = seg in padded)
    p1 = pts[seg + 1]  # [T, C]
    p2 = pts[seg + 2]  # [T, C]
    p3 = pts[seg + 3]  # [T, C]

    tt = frac.unsqueeze(1)   # [T, 1]
    tt2 = tt * tt
    tt3 = tt2 * tt

    # Catmull-Rom basis (tau = 0.5)
    out = 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * tt
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * tt2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * tt3
    )  # [T, C]
    return out


def _resample_1d(
    signal: torch.Tensor,  # [T, C]
    new_indices: torch.Tensor,  # [T]  float indices into dim-0
    device,
) -> torch.Tensor:
    """Linearly resample ``signal`` at fractional ``new_indices``."""
    T, C = signal.shape
    idx0 = new_indices.long().clamp(0, T - 2)
    idx1 = (idx0 + 1).clamp(0, T - 1)
    frac = (new_indices - idx0.float()).unsqueeze(1)  # [T, 1]
    return signal[idx0] * (1.0 - frac) + signal[idx1] * frac  # [T, C]


def _make_temporal_profiles(
    K: int,
    T: int,
    sigma: float,
    rng: torch.Generator,
    device,
) -> torch.Tensor:
    """Smoothed, enveloped random temporal profiles in ``[-1, 1]``.

    Returns shape ``[K, T]``.
    """
    raw = torch.randn(K, T, device=device, generator=rng)
    smoothed = _gaussian_smooth_1d(raw.T, sigma, device).T  # [K, T]
    envelope = torch.sin(torch.linspace(0, torch.pi, T, device=device))  # [T]
    smoothed = smoothed * envelope.unsqueeze(0)
    mx = smoothed.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    return smoothed / mx  # [K, T]


# ── configuration dataclasses ──────────────────────────────────────────────


@dataclass
class TrajectoryConfig:
    """Distribution parameters for stochastic camera trajectory generation.

    Each numeric field may be a **scalar** (fixed) or **``(lo, hi)``** (sampled
    on every :meth:`PseudoGTGenerator.generate` call).

    Attributes
    ----------
    n_frames : int
        Number of output frames (temporal length of the sequence).
    z_bias_range
        Starting-point Z offset.  Positive ⇒ camera starts *closer* to the
        surface (zoom-in); negative ⇒ *farther* (zoom-out).
    complexity_range
        Controls trajectory morphology on a ``[0, 1]`` scale.

        * **0** — gentle, nearly straight-line motion (high temporal
          smoothing, small rotations).
        * **1** — winding path with sharp direction changes and large
          rotations.
    translation_range
        Per-frame translation magnitude (scene-metric units, per axis).
    rotation_range_deg
        Per-frame rotation magnitude (degrees, per axis).
    forward_bias_range
        Constant +Z drift per frame (endoscopic forward-zoom feel).
    speed_scale_range
        Global speed multiplier applied to all deltas.
    still_fraction_range
        Fraction of the trajectory spent in near-still / slow-motion segments.
    """

    n_frames: int = 24

    z_bias_range: ScalarOrFloatRange = (-0.1, 0.1)
    complexity_range: ScalarOrFloatRange = (0.3, 0.7)

    translation_range: ScalarOrFloatRange = (0.02, 0.08)
    rotation_range_deg: ScalarOrFloatRange = (2.0, 8.0)
    forward_bias_range: ScalarOrFloatRange = (0.005, 0.025)

    speed_scale_range: ScalarOrFloatRange = (0.5, 1.5)
    still_fraction_range: ScalarOrFloatRange = (0.0, 0.15)


@dataclass
class DeformationConfig:
    """Distribution parameters for stochastic scene deformation.

    ``*_range`` fields follow the same scalar vs ``(lo, hi)`` convention as
    :class:`TrajectoryConfig`.

    Attributes
    ----------
    n_deformers_range
        Number of independent deformation control points.
    sigma_frac_range
        Gaussian kernel radius as a fraction of the scene's spatial extent.
    amplitude_frac_range
        Peak displacement as a fraction of the scene's spatial extent
        (used by *drag* and *inflate* deformers).
    drag_weight, inflate_weight, twist_weight : float
        Relative probabilities for assigning each deformer's primary type.
    twist_max_deg_range
        Maximum Rodrigues-rotation angle for *twist*-type deformers.
    temporal_smooth_range
        Gaussian smoothing σ (in frames) for the deformation temporal profiles.
    """

    n_deformers_range: ScalarOrIntRange = (2, 5)

    sigma_frac_range: ScalarOrFloatRange = (0.08, 0.25)
    amplitude_frac_range: ScalarOrFloatRange = (0.005, 0.03)

    drag_weight: float = 0.5
    inflate_weight: float = 0.3
    twist_weight: float = 0.2

    twist_max_deg_range: ScalarOrFloatRange = (1.0, 5.0)
    temporal_smooth_range: ScalarOrFloatRange = (2.0, 6.0)


@dataclass
class GridConfig:
    """Dense query-point grid layout.

    Attributes
    ----------
    grid_size : int
        The grid is ``grid_size × grid_size`` (total Q = grid_size²).
    margin_frac : float
        Margin from image edges as a fraction of the image dimension.
    """

    grid_size: int = 32
    margin_frac: float = 0.03


def _mapping_get(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def _coerce_scalar_or_float_range(value: Any, *, field_name: str) -> ScalarOrFloatRange:
    """Parse a YAML value into a fixed float or ``(lo, hi)`` pair."""
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(
                f"{field_name}: expected length-2 sequence [lo, hi] or a scalar, got {value!r}"
            )
        return (float(value[0]), float(value[1]))
    if isinstance(value, bool):
        raise TypeError(f"{field_name}: expected float, int, or [lo, hi], not bool")
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(
        f"{field_name}: expected float, int, or [lo, hi], got {type(value).__name__}"
    )


def _coerce_scalar_or_int_range(value: Any, *, field_name: str) -> ScalarOrIntRange:
    """Parse a YAML value into a fixed int or ``(lo, hi)`` inclusive int pair."""
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(
                f"{field_name}: expected length-2 sequence [lo, hi] or a scalar, got {value!r}"
            )
        return (int(value[0]), int(value[1]))
    if isinstance(value, bool):
        raise TypeError(f"{field_name}: expected int or [lo, hi] ints, not bool")
    if isinstance(value, (int, float)):
        return int(value)
    raise TypeError(
        f"{field_name}: expected int or [lo, hi] ints, got {type(value).__name__}"
    )


def trajectory_config_from_run_config(config: Any, *, n_frames: int) -> TrajectoryConfig:
    """Merge optional ``PSEUDO_GT_TRAJECTORY`` from run config with defaults.

    ``n_frames`` is always taken from the caller (batch temporal length ``T``),
    not from YAML, so pseudo clips match the training / val window.

    Parameters
    ----------
    config
        Flat mapping (e.g. DotMap). Reads subtree ``PSEUDO_GT_TRAJECTORY`` if set.
    n_frames
        Output trajectory length (typically ``TRACKING_SEQUENCE_LENGTH``).
    """
    base = TrajectoryConfig(n_frames=int(n_frames))
    sec = _mapping_get(config, "PSEUDO_GT_TRAJECTORY")
    if sec is None:
        return base
    overrides: dict = {}
    for fname in (
        "z_bias_range",
        "complexity_range",
        "translation_range",
        "rotation_range_deg",
        "forward_bias_range",
        "speed_scale_range",
        "still_fraction_range",
    ):
        raw = _mapping_get(sec, fname)
        if raw is None:
            continue
        overrides[fname] = _coerce_scalar_or_float_range(raw, field_name=fname)
    return replace(base, **overrides) if overrides else base


def deformation_config_from_run_config(config: Any) -> DeformationConfig:
    """Merge optional ``PSEUDO_GT_DEFORMATION`` from run config with defaults."""
    base = DeformationConfig()
    sec = _mapping_get(config, "PSEUDO_GT_DEFORMATION")
    if sec is None:
        return base
    overrides: dict = {}
    raw_nd = _mapping_get(sec, "n_deformers_range")
    if raw_nd is not None:
        overrides["n_deformers_range"] = _coerce_scalar_or_int_range(
            raw_nd, field_name="n_deformers_range",
        )
    for fname in (
        "sigma_frac_range",
        "amplitude_frac_range",
        "twist_max_deg_range",
        "temporal_smooth_range",
    ):
        raw = _mapping_get(sec, fname)
        if raw is None:
            continue
        overrides[fname] = _coerce_scalar_or_float_range(raw, field_name=fname)
    for fname in ("drag_weight", "inflate_weight", "twist_weight"):
        raw = _mapping_get(sec, fname)
        if raw is None:
            continue
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TypeError(f"{fname}: expected float, got {type(raw).__name__}")
        overrides[fname] = float(raw)
    return replace(base, **overrides) if overrides else base


@dataclass
class PseudoGTResult:
    """Complete output of a single pseudo ground-truth generation call.

    All tensors live on the same device that was used during generation.
    """

    # core tracking data
    frames: torch.Tensor  # [T, 3, H, W]
    tracks: torch.Tensor  # [T, Q, 2]
    visibility: torch.Tensor  # [T, Q]  bool
    query_pixels: torch.Tensor  # [Q, 2]
    # Novel-view RGB validity from Project holemask (1 = reliable, 0 = hole / inpaint)
    frame_valid: torch.Tensor  # [T, 1, H, W] float in [0, 1]

    # camera
    poses: torch.Tensor  # [T, 4, 4]
    intrinsics: torch.Tensor  # [1, 3, 3]

    # source data
    source_image: torch.Tensor  # [1, 3, H, W]
    depth: torch.Tensor  # [1, 1, H, W]

    # 3-D data (for visualisation)
    cloud_ref: torch.Tensor  # [1, 4, N]
    cloud_rgb: torch.Tensor  # [1, 3, N]
    query_3d_ref: torch.Tensor  # [1, 4, Q]
    clouds_deformed: List[torch.Tensor]  # T × [1, 4, N]
    queries_deformed: List[torch.Tensor]  # T × [1, 4, Q]

    # spatial dimensions
    height: int
    width: int

    zbuf_debug: Optional[torch.Tensor] = None  # [T, H, W, 3] uint8 RGB


# ── generator ──────────────────────────────────────────────────────────────


class PseudoGTGenerator:
    """One-call pseudo ground-truth track generator.

    Instantiate once (sets up ``BackProject`` / ``Project`` buffers for a
    given resolution), then call :meth:`generate` with different images and
    configs.  Use ``sys.path`` so this repo is imported before any
    ``site-packages`` copy of ``gatetracker``.
    """

    def __init__(self, height: int, width: int, device: str = "cuda"):
        self.height = height
        self.width = width
        self.device = device
        self._backproject = BackProject(height, width).to(device)
        self._project = Project(height, width).to(device)

    # ── public API ──────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
        intrinsics: torch.Tensor,
        trajectory: Optional[TrajectoryConfig] = None,
        deformation: Optional[DeformationConfig] = None,
        grid: Optional[GridConfig] = None,
        seed: Optional[int] = None,
        randomize_trajectory: bool = True,
        visibility_z_tol_frac: float = 0.08,
        visibility_z_abs_min: float = 1e-3,
        visibility_depth_dilate: int = 5,
        visibility_query_patch_rad: int = 1,
        visibility_temporal_window: int = 3,
        frame_valid_erode_px: int = 0,
    ) -> PseudoGTResult:
        """Generate pseudo-GT tracks from a single image + geometry.

        Parameters
        ----------
        image : Tensor [1, 3, H, W]
            Source RGB image normalised to ``[0, 1]``.
        depth : Tensor [1, 1, H, W]
            Metric depth (**not** normalised to ``[0, 1]``).
        intrinsics : Tensor [1, 3, 3]
            Camera intrinsic matrix.
        trajectory : TrajectoryConfig, optional
        deformation : DeformationConfig, optional
        grid : GridConfig, optional
        seed : int, optional
            Seeds the main RNG (deformation, etc.).  If *None*, non-deterministic.
        randomize_trajectory : bool, default ``True``
            If *True*, camera path uses a separate RNG with fresh entropy each
            call (different path even when ``seed`` is fixed).  If *False*, the
            trajectory shares the main RNG (fully reproducible with ``seed``).
        visibility_z_tol_frac, visibility_z_abs_min
            Relative and absolute depth tolerances for z-buffer visibility.
        visibility_depth_dilate
            Odd kernel size for min-pooling the splatted depth map.
        visibility_query_patch_rad
            Pixel patch radius when reading the z-buffer at each query.
        visibility_temporal_window
            Odd temporal max-pool width on visibility; ``1`` disables.
        frame_valid_erode_px
            Erosion radius (pixels) on ``frame_valid`` to shrink valid regions away
            from splat / inpaint boundaries; ``0`` disables.

        Returns
        -------
        PseudoGTResult
        """
        trajectory = trajectory or TrajectoryConfig()
        deformation = deformation or DeformationConfig()
        grid = grid or GridConfig()

        dev = self.device
        H, W = self.height, self.width
        T = trajectory.n_frames

        rng = torch.Generator(device=dev)
        if seed is not None:
            rng.manual_seed(seed)
        else:
            rng.seed()

        if randomize_trajectory:
            rng_traj = torch.Generator(device=dev)
            rng_traj.seed()
        else:
            rng_traj = rng

        invK = torch.inverse(intrinsics)  # [1, 3, 3]

        # ── back-project scene cloud + query grid ───────────────────────
        bp = self._backproject(image, depth, invK)
        X0 = bp["xyz1"]  # [1, 4, H*W]
        C0 = bp["rgb"]  # [1, 3, H*W]

        margin = max(1, int(min(H, W) * grid.margin_frac))
        gx = torch.linspace(margin, W - margin, grid.grid_size, device=dev)
        gy = torch.linspace(margin, H - margin, grid.grid_size, device=dev)
        gy_g, gx_g = torch.meshgrid(gy, gx, indexing="ij")
        Q0_px = torch.stack(
            [gx_g.reshape(-1), gy_g.reshape(-1)], dim=-1
        ).unsqueeze(0)  # [1, Q, 2]

        bp_q = self._backproject(image, depth, invK, points_match=Q0_px)
        Q0_3d = bp_q["points_match_3d"]  # [1, 4, Q]

        # ── trajectory + deformation ────────────────────────────────────
        poses = self._build_trajectory(trajectory, rng_traj, dev)  # [T, 4, 4]
        deform_fn = self._build_deformation(
            deformation, X0[:, :3, :], T, rng, dev
        )

        # ── render loop ─────────────────────────────────────────────────
        all_frames: List[torch.Tensor] = []
        all_tracks: List[torch.Tensor] = []
        all_vis: List[torch.Tensor] = []
        all_X: List[torch.Tensor] = []
        all_Q: List[torch.Tensor] = []
        all_zbuf_dbg: List[torch.Tensor] = []
        all_frame_valid: List[torch.Tensor] = []

        for t in range(T):
            Xt = deform_fn(X0, t)  # [1, 4, H*W]
            Qt = deform_fn(Q0_3d, t)  # [1, 4, Q]

            Pt = poses[t].unsqueeze(0)  # [1, 4, 4]

            proj_out = self._project(
                Xt, C0, intrinsics, Pt,
                points_match_3d=Qt,
                return_mask=True,
                median_kernel_size=5,
            )

            Pt_inv = torch.inverse(Pt)
            Xt_cam = torch.bmm(Pt_inv, Xt)  # [1, 4, N]
            Qt_cam = torch.bmm(Pt_inv, Qt)  # [1, 4, Q]
            _, vis_t, zbuf_dbg_t = self._compute_visibility(
                Xt_cam,
                Qt_cam,
                intrinsics,
                H,
                W,
                z_tol_frac=visibility_z_tol_frac,
                z_abs_min=visibility_z_abs_min,
                dilate_k=visibility_depth_dilate,
                query_patch_rad=visibility_query_patch_rad,
            )

            all_frames.append(proj_out["warped"])
            all_tracks.append(proj_out["matches"])
            hm = proj_out.get("mask")
            if hm is None:
                fv = torch.ones(1, 1, H, W, device=dev, dtype=torch.float32)
            else:
                fv = hm[:, :1, :, :].to(dtype=torch.float32).clamp(0.0, 1.0)
            all_frame_valid.append(fv)
            all_vis.append(vis_t)
            all_X.append(Xt)
            all_Q.append(Qt)
            all_zbuf_dbg.append(zbuf_dbg_t)

        vis_stack = torch.cat(all_vis, dim=0)  # [T, Q]
        if visibility_temporal_window > 1:
            vis_stack = PseudoGTGenerator._smooth_visibility_temporal(
                vis_stack, visibility_temporal_window
            )

        frame_valid_stack = torch.cat(all_frame_valid, dim=0)  # [T, 1, H, W]
        frame_valid_stack = PseudoGTGenerator._erode_valid_mask_2d(
            frame_valid_stack, frame_valid_erode_px
        )

        return PseudoGTResult(
            frames=torch.cat(all_frames, dim=0),       # [T, 3, H, W]
            tracks=torch.cat(all_tracks, dim=0),        # [T, Q, 2]
            visibility=vis_stack,                        # [T, Q]
            query_pixels=Q0_px.squeeze(0),               # [Q, 2]
            frame_valid=frame_valid_stack,
            poses=poses,
            intrinsics=intrinsics,
            source_image=image,
            depth=depth,
            cloud_ref=X0,
            cloud_rgb=C0,
            query_3d_ref=Q0_3d,
            clouds_deformed=all_X,
            queries_deformed=all_Q,
            height=H,
            width=W,
            zbuf_debug=torch.cat(all_zbuf_dbg, dim=0),  # [T, H, W, 3]
        )

    # ── trajectory ──────────────────────────────────────────────────────

    @staticmethod
    def _build_trajectory(
        cfg: TrajectoryConfig,
        rng: torch.Generator,
        device,
    ) -> torch.Tensor:
        """Sample a smooth random SE(3) camera trajectory.

        Uses a **waypoint-spline** strategy: sample ``n_waypoints`` random
        6-DoF poses, then interpolate with cubic B-spline to ``T`` frames.
        The *complexity* knob controls how many waypoints (more = more
        direction changes and rotations).

        Returns ``[T, 4, 4]`` cumulative poses (frame-0 includes only the
        *z_bias* offset).
        """
        T = cfg.n_frames

        complexity = _sample_f(rng, cfg.complexity_range, device)
        z_bias = _sample_f(rng, cfg.z_bias_range, device)
        trans_ampl = _sample_f(rng, cfg.translation_range, device)
        rot_ampl_deg = _sample_f(rng, cfg.rotation_range_deg, device)
        rot_ampl = torch.deg2rad(torch.tensor(rot_ampl_deg, device=device))
        fwd_bias = _sample_f(rng, cfg.forward_bias_range, device)
        spd_scale = _sample_f(rng, cfg.speed_scale_range, device)
        still_frac = _sample_f(rng, cfg.still_fraction_range, device)

        rot_scale = 0.3 + 0.9 * complexity  # 0→0.3x  1→1.2x

        # per-axis random scale  [3]
        trans_ax = trans_ampl * (
            0.5 + torch.rand(3, device=device, generator=rng)
        )
        rot_ax = rot_ampl * rot_scale * (
            0.5 + torch.rand(3, device=device, generator=rng)
        )

        # number of waypoints driven by complexity: 2 (gentle) .. ~T/3 (winding)
        n_wp = max(2, int(2 + (T / 3 - 2) * complexity))

        # random waypoint poses in 6-DoF (translation + euler)
        wp = torch.randn(n_wp, 6, device=device, generator=rng)  # [n_wp, 6]
        wp[:, :3] *= trans_ax * spd_scale
        wp[:, 3:] *= rot_ax
        wp[:, 2] += fwd_bias * T / max(n_wp, 1)

        # cumulative sum so waypoints form a path
        wp = torch.cumsum(wp, dim=0)  # [n_wp, 6]
        # prepend origin
        wp = torch.cat([torch.zeros(1, 6, device=device), wp], dim=0)  # [n_wp+1, 6]

        # cubic B-spline interpolation to T frames
        poses_6dof = _catmull_rom_interp(wp, T, device)  # [T, 6]

        # speed profile with optional still / slow segments
        speed = PseudoGTGenerator._make_speed_profile(
            T, still_frac, complexity, rng, device
        )  # [T]

        # apply speed modulation: re-parameterise arc by integrating speed
        arc = torch.cumsum(speed, dim=0)         # [T]
        arc = arc / arc[-1]                       # normalise to [0, 1]
        arc = arc * (T - 1)                       # map back to frame indices
        # re-sample poses_6dof at the new arc positions (linear interp)
        poses_6dof = _resample_1d(poses_6dof, arc, device)  # [T, 6]

        poses_6dof[0] = 0.0
        poses_6dof[:, 2] += z_bias

        return euler2mat(poses_6dof)  # [T, 4, 4]

    @staticmethod
    def _make_speed_profile(
        T: int,
        still_frac: float,
        complexity: float,
        rng: torch.Generator,
        device,
    ) -> torch.Tensor:
        """Smooth speed envelope ``[T]`` in ``(0, 1]`` with near-still dips."""
        speed = torch.ones(T, device=device)
        if still_frac < 0.01 or T < 6:
            return speed

        n_still = max(2, int(T * still_frac))
        n_seg = max(1, min(3, int(1 + 2 * complexity)))
        frames_per = max(2, n_still // n_seg)
        t_idx = torch.arange(T, device=device, dtype=torch.float32)

        for _ in range(n_seg):
            ctr = _uniform_int(rng, 2, max(3, T - 2), device)
            hw = max(frames_per // 2, 1)
            dip = torch.exp(-0.5 * ((t_idx - ctr) / hw) ** 2)
            speed = speed * (1.0 - 0.95 * dip)

        return speed.clamp(min=0.02)

    # ── deformation ─────────────────────────────────────────────────────

    @staticmethod
    def _build_deformation(
        cfg: DeformationConfig,
        cloud_xyz: torch.Tensor,  # [1, 3, N]
        T: int,
        rng: torch.Generator,
        device,
    ):
        """Build a deformation closure ``fn(X, t) -> X_def``.

        The returned callable accepts ``X  [1, 3or4, N]`` and an integer
        frame index ``t`` and returns a deformed copy.
        """
        K = _sample_i(rng, cfg.n_deformers_range, device)
        if K == 0:
            return lambda X, t: X  # identity

        xyz = cloud_xyz[0]  # [3, N]
        N = xyz.shape[1]
        centroid = xyz.mean(dim=1, keepdim=True)  # [3, 1]
        extent = torch.quantile((xyz - centroid).norm(dim=0), 0.9).item()

        # control-point centres sampled from the cloud
        idx = torch.randint(0, N, (K,), device=device, generator=rng)
        centres = xyz[:, idx].T  # [K, 3]

        # per-deformer parameters ------------------------------------------------
        sigmas = torch.tensor(
            [extent * _sample_f(rng, cfg.sigma_frac_range, device) for _ in range(K)],
            device=device,
        )  # [K]
        amplitudes = torch.tensor(
            [extent * _sample_f(rng, cfg.amplitude_frac_range, device) for _ in range(K)],
            device=device,
        )  # [K]

        # type assignment via weighted categorical  (0=drag, 1=inflate, 2=twist)
        total_w = cfg.drag_weight + cfg.inflate_weight + cfg.twist_weight
        p_drag = cfg.drag_weight / total_w
        p_infl = cfg.inflate_weight / total_w
        types = torch.zeros(K, dtype=torch.long, device=device)
        for k in range(K):
            r = torch.rand(1, device=device, generator=rng).item()
            if r < p_drag:
                types[k] = 0
            elif r < p_drag + p_infl:
                types[k] = 1
            else:
                types[k] = 2

        drag_dir = F.normalize(
            torch.randn(K, 3, device=device, generator=rng), dim=1
        )  # [K, 3]

        twist_axis = F.normalize(
            torch.randn(K, 3, device=device, generator=rng), dim=1
        )  # [K, 3]
        twist_max = torch.deg2rad(
            torch.tensor(
                [_sample_f(rng, cfg.twist_max_deg_range, device) for _ in range(K)],
                device=device,
            )
        )  # [K]

        t_smooth = _sample_f(rng, cfg.temporal_smooth_range, device)
        profiles = _make_temporal_profiles(K, T, t_smooth, rng, device)  # [K, T]

        # ── closure ----------------------------------------------------------

        def _deform(X: torch.Tensor, t: int) -> torch.Tensor:
            has_homo = X.shape[1] == 4
            pts = X[:, :3, :]  # [1, 3, M]
            M = pts.shape[2]
            prof = profiles[:, t]  # [K]

            diff = pts[0].T.unsqueeze(0) - centres.unsqueeze(1)  # [K, M, 3]
            dist_sq = (diff ** 2).sum(dim=2)  # [K, M]
            w = torch.exp(
                -dist_sq / (2.0 * sigmas.unsqueeze(1) ** 2)
            )  # [K, M]

            disp = torch.zeros(1, 3, M, device=device)

            for k in range(K):
                scale_k = w[k] * prof[k] * amplitudes[k]  # [M]

                if types[k] == 0:  # drag
                    disp = disp + (
                        scale_k.unsqueeze(0) * drag_dir[k].unsqueeze(1)
                    ).unsqueeze(0)  # [1, 3, M]

                elif types[k] == 1:  # inflate / deflate
                    radial = F.normalize(diff[k], dim=1)  # [M, 3]
                    disp = disp + (
                        scale_k.unsqueeze(1) * radial
                    ).T.unsqueeze(0)  # [1, 3, M]

                else:  # twist (Rodrigues)
                    angle_k = prof[k] * twist_max[k]
                    if angle_k.abs() < 1e-7:
                        continue
                    ax = twist_axis[k]  # [3]
                    d = diff[k]  # [M, 3]
                    ca, sa = torch.cos(angle_k), torch.sin(angle_k)
                    cross = torch.stack([
                        ax[1] * d[:, 2] - ax[2] * d[:, 1],
                        ax[2] * d[:, 0] - ax[0] * d[:, 2],
                        ax[0] * d[:, 1] - ax[1] * d[:, 0],
                    ], dim=1)  # [M, 3]
                    dot = (ax.unsqueeze(0) * d).sum(1, keepdim=True)  # [M, 1]
                    rot_d = d * ca + cross * sa + ax * dot * (1 - ca)
                    tw = (rot_d - d) * w[k].unsqueeze(1)  # [M, 3]
                    disp = disp + tw.T.unsqueeze(0)

            out = pts + disp
            if has_homo:
                return torch.cat([out, X[:, 3:4, :]], dim=1)
            return out

        return _deform

    # ── validity (novel-view holes) ─────────────────────────────────────

    @staticmethod
    def _erode_valid_mask_2d(
        valid: torch.Tensor,  # [T, 1, H, W] float 0..1
        erode_px: int,
    ) -> torch.Tensor:
        """Erode valid regions by dilating the invalid set (max-pool on 1 - valid)."""
        if erode_px is None or int(erode_px) <= 0:
            return valid
        k = 2 * int(erode_px) + 1
        inv = 1.0 - valid
        inv_d = F.max_pool2d(inv, kernel_size=k, stride=1, padding=int(erode_px))
        return (1.0 - inv_d).clamp(0.0, 1.0)

    # ── visibility ──────────────────────────────────────────────────────

    @staticmethod
    def _query_neighborhood_zbuf_min(
        zbuf_flat: torch.Tensor,  # [1, H * W]
        H: int,
        W: int,
        u: torch.Tensor,  # [1, Q]  long
        v: torch.Tensor,  # [1, Q]  long
        rad: int,
    ) -> torch.Tensor:
        """Min dilated z-buffer over a ``(2*rad+1)²`` patch; returns ``[1, Q]``."""
        if rad <= 0:
            return zbuf_flat.gather(1, v * W + u)
        dev = u.device
        offs = torch.arange(-rad, rad + 1, device=dev, dtype=torch.long)
        uu = u.unsqueeze(2).unsqueeze(3) + offs.view(1, 1, -1, 1)
        vv = v.unsqueeze(2).unsqueeze(3) + offs.view(1, 1, 1, -1)
        uu = (uu + torch.zeros_like(vv)).clamp(0, W - 1)
        vv = (vv + torch.zeros_like(uu)).clamp(0, H - 1)
        lin = (vv * W + uu).reshape(1, -1)
        z_nb = zbuf_flat.gather(1, lin).view(1, u.shape[1], -1)
        z_nb = z_nb.masked_fill(torch.isinf(z_nb), float("inf"))
        return z_nb.min(dim=-1).values

    @staticmethod
    def _smooth_visibility_temporal(vis: torch.Tensor, window: int) -> torch.Tensor:
        """OR-pool visibility over ``window`` frames (odd), ``[T, Q]``."""
        if window <= 1 or vis.shape[0] < 2:
            return vis
        if window % 2 == 0:
            window = window + 1
        w = window // 2
        x = vis.float().permute(1, 0).unsqueeze(1)  # [Q, 1, T]
        x = F.pad(x, (w, w), mode="replicate")
        y = F.max_pool1d(x, kernel_size=window, stride=1, padding=0)
        return y.squeeze(1).permute(1, 0) > 0.5

    @staticmethod
    def _compute_visibility(
        cloud_cam: torch.Tensor,  # [1, 4, N]
        query_cam: torch.Tensor,  # [1, 4, Q]
        K: torch.Tensor,  # [1, 3, 3]
        H: int,
        W: int,
        z_tol_frac: float = 0.08,
        z_abs_min: float = 1e-3,
        dilate_k: int = 5,
        query_patch_rad: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Float z-buffer visibility + debug RGB overlay ``[1, H, W, 3]`` uint8."""
        dev = cloud_cam.device
        B = 1

        proj_c = torch.bmm(K, cloud_cam[:, :3, :])  # [1, 3, N]
        uv_c = proj_c[:, :2, :] / proj_c[:, 2:3, :].clamp(min=1e-6)
        z_c = cloud_cam[:, 2, :]  # [1, N]

        u_c = uv_c[:, 0, :].round().long().clamp(0, W - 1)
        v_c = uv_c[:, 1, :].round().long().clamp(0, H - 1)
        lin_c = v_c * W + u_c  # [1, N]

        zbuf_min = torch.full((B, H * W), float("inf"), device=dev)
        zbuf_min.scatter_reduce_(1, lin_c, z_c, reduce="amin", include_self=True)

        zbuf_max = torch.full((B, H * W), 0.0, device=dev)
        zbuf_max.scatter_reduce_(1, lin_c, z_c, reduce="amax", include_self=True)
        hit_count = torch.zeros(B, H * W, device=dev)
        hit_count.scatter_reduce_(
            1, lin_c, torch.ones_like(z_c), reduce="sum", include_self=False
        )

        gap_mask = hit_count < 0.5
        spread = (zbuf_max - zbuf_min).abs()
        med_z = zbuf_min.clamp(min=1e-6)
        conflict_mask = (~gap_mask) & (spread > z_tol_frac * med_z)

        dbg = torch.zeros(B, H * W, 3, device=dev, dtype=torch.uint8)
        dbg[gap_mask] = torch.tensor([255, 0, 0], device=dev, dtype=torch.uint8)
        dbg[conflict_mask] = torch.tensor(
            [255, 220, 0], device=dev, dtype=torch.uint8
        )
        dbg[(~gap_mask) & (~conflict_mask)] = torch.tensor(
            [0, 200, 0], device=dev, dtype=torch.uint8
        )
        zbuf_debug = dbg.view(B, H, W, 3)

        dilate_k = max(int(dilate_k), 1)
        if dilate_k % 2 == 0:
            dilate_k += 1
        zbuf_2d = zbuf_min.view(B, 1, H, W)
        pad = dilate_k // 2
        zbuf_2d = -F.max_pool2d(
            -zbuf_2d, kernel_size=dilate_k, stride=1, padding=pad
        )
        zbuf = zbuf_2d.view(B, H * W)

        proj_q = torch.bmm(K, query_cam[:, :3, :])  # [1, 3, Q]
        uv_q = proj_q[:, :2, :] / proj_q[:, 2:3, :].clamp(min=1e-6)
        z_q = query_cam[:, 2, :]  # [1, Q]

        u_q = uv_q[:, 0, :].round().long().clamp(0, W - 1)
        v_q = uv_q[:, 1, :].round().long().clamp(0, H - 1)
        zbuf_at_q = PseudoGTGenerator._query_neighborhood_zbuf_min(
            zbuf, H, W, u_q, v_q, query_patch_rad
        )

        in_bounds = (
            (uv_q[:, 0, :] >= 0)
            & (uv_q[:, 0, :] < W)
            & (uv_q[:, 1, :] >= 0)
            & (uv_q[:, 1, :] < H)
        )
        z_tol = torch.maximum(
            z_tol_frac * z_q.abs(),
            z_q.new_tensor(z_abs_min),
        )
        visible = in_bounds & (z_q > 0) & (z_q <= zbuf_at_q + z_tol)

        return uv_q.permute(0, 2, 1), visible, zbuf_debug

    # ── visualisation helpers ───────────────────────────────────────────

    @staticmethod
    def log_to_rerun(
        result: PseudoGTResult,
        *,
        subsample_cloud: int = 8,
        n_tracks_3d: int = 128,
    ) -> None:
        """Log a complete :class:`PseudoGTResult` to an **already initialised**
        Rerun recording.

        Parameters
        ----------
        result : PseudoGTResult
        subsample_cloud : int
            Point-cloud subsampling factor for the per-frame 3-D cloud.
        n_tracks_3d : int
            Number of query trajectories to draw as 3-D polylines.
        """
        import rerun as rr
        from matplotlib import colormaps

        T = result.frames.shape[0]
        Q = result.tracks.shape[1]
        H, W = result.height, result.width
        K_np = result.intrinsics[0].cpu().numpy()

        cmap = colormaps["hsv"]
        qcolors = (
            cmap(np.linspace(0, 1, Q, endpoint=False))[:, :3] * 255
        ).astype(np.uint8)  # [Q, 3]

        # static: pinhole
        rr.log(
            "world/camera",
            rr.Pinhole(image_from_camera=K_np, width=W, height=H),
            static=True,
        )

        # static: camera path
        cam_xyz = (
            torch.stack([result.poses[t, :3, 3] for t in range(T)])
            .cpu()
            .numpy()
        )  # [T, 3]
        rr.log(
            "world/cam_path",
            rr.LineStrips3D([cam_xyz], colors=[[0, 200, 255]]),
            static=True,
        )

        # static: 3-D track polylines (subsampled)
        show_idx = np.linspace(0, Q - 1, min(n_tracks_3d, Q), dtype=int)
        for qi in show_idx:
            pts = (
                torch.stack([result.queries_deformed[t][0, :3, qi] for t in range(T)])
                .cpu()
                .numpy()
            )  # [T, 3]
            rr.log(
                f"world/tracks3d/{qi}",
                rr.LineStrips3D([pts], colors=[qcolors[qi].tolist()]),
                static=True,
            )

        # per-frame
        sub = subsample_cloud
        for t in range(T):
            rr.set_time_sequence("frame", t)

            Xt = result.clouds_deformed[t][0, :3, ::sub].T.cpu().numpy()
            Ct = (
                (result.cloud_rgb[0, :, ::sub].T.cpu().numpy() * 255)
                .clip(0, 255)
                .astype(np.uint8)
            )
            rr.log("world/cloud", rr.Points3D(Xt, colors=Ct, radii=0.002))

            P = result.poses[t].cpu().numpy()
            rr.log(
                "world/camera",
                rr.Transform3D(translation=P[:3, 3], mat3x3=P[:3, :3]),
            )

            img = (
                (result.frames[t].permute(1, 2, 0).cpu().numpy() * 255)
                .clip(0, 255)
                .astype(np.uint8)
            )
            rr.log("world/camera/image", rr.Image(img))

            uv = result.tracks[t].cpu().numpy()  # [Q, 2]
            vis = result.visibility[t].cpu().numpy()  # [Q]
            vc = qcolors.copy()
            vc[~vis] = [60, 60, 60]
            rr.log(
                "world/camera/tracks2d",
                rr.Points2D(uv, colors=vc, radii=2.0),
            )

    @staticmethod
    def render_video(
        result: PseudoGTResult,
        path: str,
        *,
        fps: int = 8,
        trail_length: int = 8,
        log_to_rerun: bool = True,
        zbuf_overlay_alpha: float = 0.0,
    ) -> None:
        """Write an MP4 video with track dots and coloured trails.

        Optionally logs each annotated frame to the active Rerun recording
        under ``video/annotated``.

        Parameters
        ----------
        result : PseudoGTResult
        path : str
            Output ``.mp4`` file path.
        fps : int
        trail_length : int
            How many past frames to include in the polyline trail.
        log_to_rerun : bool
            Also log annotated frames to Rerun if a recording is active.
        zbuf_overlay_alpha : float
            Blend ``[0, 1]`` for z-buffer debug overlay; ``0`` = off.  When
            ``> 0``, output is side-by-side: tracks | tracks + overlay.
        """
        import cv2
        from matplotlib import colormaps

        T = result.frames.shape[0]
        Q = result.tracks.shape[1]
        H, W = result.height, result.width

        cmap = colormaps["hsv"]
        colors_bgr = (
            cmap(np.linspace(0, 1, Q, endpoint=False))[:, :3] * 255
        ).astype(np.uint8)[:, ::-1].copy()

        import tempfile, subprocess, shutil
        _has_ffmpeg = shutil.which("ffmpeg") is not None
        tmp_path = tempfile.mktemp(suffix=".mp4") if _has_ffmpeg else path

        alpha = float(np.clip(zbuf_overlay_alpha, 0.0, 1.0))
        has_zbuf = alpha > 0 and result.zbuf_debug is not None
        if has_zbuf:
            zbuf_np = result.zbuf_debug.cpu().numpy()

        out_w = W * 2 if has_zbuf else W
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_path, fourcc, fps, (out_w, H))

        trk = result.tracks.cpu().numpy().astype(np.int32)  # [T, Q, 2]
        vis = result.visibility.cpu().numpy()  # [T, Q]

        try:
            import rerun as rr
            has_rr = log_to_rerun
        except ImportError:
            has_rr = False

        for t in range(T):
            img = (
                (result.frames[t].permute(1, 2, 0).cpu().numpy() * 255)
                .clip(0, 255)
                .astype(np.uint8)
            )
            canvas = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            t0 = max(0, t - trail_length)
            for qi in range(Q):
                v_win = vis[t0 : t + 1, qi]
                if v_win.sum() < 2:
                    continue
                pts = trk[t0 : t + 1, qi][v_win]
                color = tuple(int(c) for c in colors_bgr[qi])
                cv2.polylines(
                    canvas, [pts], False, color, 1, cv2.LINE_AA
                )

            vis_idx = np.where(vis[t])[0]
            occ_idx = np.where(~vis[t])[0]
            for qi in vis_idx:
                cv2.circle(
                    canvas, tuple(trk[t, qi]), 3,
                    tuple(int(c) for c in colors_bgr[qi]),
                    -1, cv2.LINE_AA,
                )
            for qi in occ_idx:
                cv2.circle(
                    canvas, tuple(trk[t, qi]), 2,
                    (80, 80, 80), -1, cv2.LINE_AA,
                )

            cv2.putText(
                canvas, f"t={t:03d}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
            )

            if has_zbuf:
                overlay_rgb = zbuf_np[t]
                overlay_bgr = overlay_rgb[:, :, ::-1].copy()
                blended = cv2.addWeighted(
                    canvas, 1.0, overlay_bgr, alpha, 0.0
                )
                cv2.putText(
                    blended, "zbuf", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                )
                frame_out = np.concatenate([canvas, blended], axis=1)
            else:
                frame_out = canvas

            writer.write(frame_out)

            if has_rr:
                rr.set_time_sequence("frame", t)
                rr.log(
                    "video/annotated",
                    rr.Image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)),
                )
                if has_zbuf:
                    rr.log("video/zbuf_debug", rr.Image(overlay_rgb))

        writer.release()

        if _has_ffmpeg:
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_path,
                 "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", "-crf", "23", path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                check=True,
            )
            os.remove(tmp_path)
