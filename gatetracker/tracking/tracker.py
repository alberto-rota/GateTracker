"""
TemporalTracker — high-level wrapper combining a pretrained Matcher (frozen
descriptor backbone) with a TemporalRefinementNetwork for long-term point
tracking.

The tracker follows a two-stage TAPIR-style pipeline:
  Stage 1: Global coarse matching using pretrained descriptors (no new params).
  Stage 2: Iterative temporal refinement using local correlation + temporal
           convolutions (trainable TemporalRefinementNetwork).
"""

import torch
import torch.nn as nn

from gatetracker.matching import correspondence
from gatetracker.matching.matcher import Matcher
from gatetracker.tracking.temporal_refinement import TemporalRefinementNetwork
from gatetracker.utils.tensor_ops import embedding2chw, chw2embedding


class TemporalTracker(nn.Module):
    """Wraps a frozen Matcher with a trainable TemporalRefinementNetwork.

    Usage::

        tracker = TemporalTracker.from_config(matcher, config)
        out = tracker.track(query_points, frames)
        tracks = out["tracks"]       # [B, Q, T, 2]
        visibility = out["visibility"]  # [B, Q, T]
    """

    def __init__(
        self,
        matcher: Matcher,
        refinement_net: TemporalRefinementNetwork,
        freeze_matcher: bool = True,
    ):
        super().__init__()
        self.matcher = matcher
        self.refinement_net = refinement_net
        self._freeze_matcher = freeze_matcher

        if freeze_matcher:
            self._freeze_matcher_params()

    def _freeze_matcher_params(self):
        """Freeze all parameters in the pretrained matcher."""
        for p in self.matcher.model.parameters():
            p.requires_grad = False
        self.matcher.model.eval()

    @property
    def device(self):
        return next(self.refinement_net.parameters()).device

    @staticmethod
    def from_config(matcher: Matcher, config) -> "TemporalTracker":
        """Instantiate a TemporalTracker from a Matcher and config dict.

        Config keys consumed:
            TEMPORAL_REFINEMENT_HIDDEN_DIM (128)
            TEMPORAL_REFINEMENT_NUM_BLOCKS (4)
            TEMPORAL_REFINEMENT_NUM_ITERS  (4)
            TEMPORAL_REFINEMENT_CORR_RADIUS (3)
            TEMPORAL_REFINEMENT_KERNEL_SIZE (5)
            FINE_REFINEMENT_DIM (64)
        """
        feature_dim = int(config.get("FINE_REFINEMENT_DIM", 64))
        refinement_net = TemporalRefinementNetwork(
            feature_dim=feature_dim,
            hidden_dim=int(config.get("TEMPORAL_REFINEMENT_HIDDEN_DIM", 128)),
            num_blocks=int(config.get("TEMPORAL_REFINEMENT_NUM_BLOCKS", 4)),
            num_iters=int(config.get("TEMPORAL_REFINEMENT_NUM_ITERS", 4)),
            corr_radius=int(config.get("TEMPORAL_REFINEMENT_CORR_RADIUS", 3)),
            kernel_size=int(config.get("TEMPORAL_REFINEMENT_KERNEL_SIZE", 5)),
        ).to(matcher.device)

        return TemporalTracker(matcher, refinement_net, freeze_matcher=True)

    # ------------------------------------------------------------------
    # Feature extraction helpers (frozen backbone)
    # ------------------------------------------------------------------

    def _encode_all_frames(self, frames: torch.Tensor, chunk_size: int = 16):
        """Encode all frames in a window through the frozen backbone.

        Processes frames in chunks of ``chunk_size`` to avoid OOM while
        still being significantly faster than a per-frame loop.

        Args:
            frames:     [B, T, 3, H, W]
            chunk_size: Max images per backbone forward pass.

        Returns:
            descriptor_maps: list of T tensors [B, C, H_p, W_p]
            fine_feature_maps: [B, T, C_f, H_f, W_f] or None
        """
        B, T = frames.shape[:2]
        flat_frames = frames.reshape(B * T, *frames.shape[2:])  # [B*T, 3, H, W]
        N_total = B * T

        desc_chunks = []
        fine_chunks = []

        with torch.no_grad():
            for start in range(0, N_total, chunk_size):
                chunk = flat_frames[start : start + chunk_size]  # [C_bs, 3, H, W]
                _, matched_tokens, _, _, fine_map, _ = (
                    self.matcher.model._encode_image(chunk)
                )
                emb = matched_tokens.permute(0, 2, 1)  # [C_bs, C, N]
                if self.matcher.model.resampled_patch_size != 16:
                    emb = embedding2chw(emb, embed_dim_last=False)
                    emb = self.matcher.model.patchsize_resampler(emb)
                    emb = chw2embedding(emb)
                desc_chunks.append(embedding2chw(emb, embed_dim_last=False))
                fine_chunks.append(fine_map)

        desc_all = torch.cat(desc_chunks, dim=0)  # [B*T, C, H_p, W_p]
        desc_per_frame = desc_all.reshape(B, T, *desc_all.shape[1:])
        descriptor_maps = [desc_per_frame[:, t] for t in range(T)]

        if fine_chunks[0] is not None:
            fine_all = torch.cat(fine_chunks, dim=0)  # [B*T, C_f, H_f, W_f]
            fine_feature_maps = fine_all.reshape(B, T, *fine_all.shape[1:])
        else:
            fine_feature_maps = None

        return descriptor_maps, fine_feature_maps

    # ------------------------------------------------------------------
    # Stage 1: Global coarse matching
    # ------------------------------------------------------------------

    def _global_coarse_match(
        self,
        query_points: torch.Tensor,
        descriptor_maps: list[torch.Tensor],
    ) -> torch.Tensor:
        """Match query descriptors from t=0 to every frame independently.

        Args:
            query_points:   [B, Q, 2] positions at frame 0.
            descriptor_maps: list of T tensors [B, C, H_p, W_p].

        Returns:
            coarse_tracks: [B, Q, T, 2] patch-center coordinates.
        """
        patch_size = self.matcher.patch_size
        T = len(descriptor_maps)

        # Extract query descriptors from frame 0
        query_desc = correspondence.sample_embeddings_at_points(
            descriptor_maps[0], query_points, patch_size,
        )  # [B, Q, C]

        coarse_positions = []
        for t in range(T):
            target_emb = descriptor_maps[t]
            B, C, H_p, W_p = target_emb.shape
            target_flat = target_emb.reshape(B, C, H_p * W_p)  # [B, C, N]
            coarse_pos = correspondence.query_to_target_coarse(
                query_desc, target_flat, patch_size,
            )  # [B, Q, 2]
            coarse_positions.append(coarse_pos)

        return torch.stack(coarse_positions, dim=2)  # [B, Q, T, 2]

    # ------------------------------------------------------------------
    # Full tracking pipeline
    # ------------------------------------------------------------------

    def track(
        self,
        query_points: torch.Tensor,
        frames: torch.Tensor,
    ) -> dict:
        """Track query points across a temporal window.

        Args:
            query_points: [B, Q, 2] pixel coordinates at frame 0.
            frames:       [B, T, 3, H, W] frame window.

        Returns:
            dict with:
                tracks:       [B, Q, T, 2] refined pixel coordinates.
                visibility:   [B, Q, T] visibility logits.
                coarse_tracks: [B, Q, T, 2] pre-refinement positions.
        """
        # Stage 1: extract features and global matching
        descriptor_maps, fine_feature_maps = self._encode_all_frames(frames)
        coarse_tracks = self._global_coarse_match(query_points, descriptor_maps)

        if fine_feature_maps is None:
            return {
                "tracks": coarse_tracks,
                "visibility": torch.ones(
                    coarse_tracks.shape[:3], device=coarse_tracks.device,
                ),
                "coarse_tracks": coarse_tracks,
            }

        # Query fine features at t=0 for the refinement network
        feature_stride = getattr(self.matcher.model, "fine_feature_stride", 4)
        query_fine = correspondence.sample_embeddings_at_points(
            fine_feature_maps[:, 0], query_points, feature_stride,
        )  # [B, Q, C_f]

        # Stage 2: iterative temporal refinement
        refined = self.refinement_net(
            coarse_tracks=coarse_tracks,
            query_features=query_fine,
            feature_maps=fine_feature_maps,
            feature_stride=feature_stride,
        )

        return {
            "tracks": refined["tracks"],           # [B, Q, T, 2]
            "visibility": refined["visibility"],    # [B, Q, T]
            "coarse_tracks": coarse_tracks,        # [B, Q, T, 2]
        }

    # ------------------------------------------------------------------
    # Sliding-window long-sequence inference
    # ------------------------------------------------------------------

    def track_long_sequence(
        self,
        query_points: torch.Tensor,
        frames: torch.Tensor,
        window_size: int = 16,
    ) -> dict:
        """Track points across a long sequence using sliding overlapping windows.

        Args:
            query_points: [B, Q, 2] initial positions at frame 0.
            frames:       [B, T_total, 3, H, W] full sequence.
            window_size:  Temporal window length.

        Returns:
            dict with tracks [B, Q, T_total, 2] and visibility [B, Q, T_total].
        """
        B, T_total = frames.shape[:2]
        Q = query_points.shape[1]
        device = frames.device
        stride = max(1, window_size // 2)

        all_tracks = torch.zeros(B, Q, T_total, 2, device=device)
        all_vis = torch.zeros(B, Q, T_total, device=device)
        weight_map = torch.zeros(B, Q, T_total, 1, device=device)

        all_tracks[:, :, 0, :] = query_points
        weight_map[:, :, 0, :] = 1.0

        for w_start in range(0, T_total - 1, stride):
            w_end = min(w_start + window_size, T_total)
            if w_end - w_start < 2:
                break

            window_frames = frames[:, w_start:w_end]  # [B, T_w, 3, H, W]

            if w_start == 0:
                qp = query_points
            else:
                qp = all_tracks[:, :, w_start, :]

            with torch.no_grad():
                out = self.track(qp, window_frames)

            window_tracks = out["tracks"]  # [B, Q, T_w, 2]
            vis = out["visibility"]  # [B, Q, T_w]
            conf = torch.sigmoid(vis).unsqueeze(-1)  # [B, Q, T_w, 1]

            for local_t in range(w_end - w_start):
                global_t = w_start + local_t
                w = conf[:, :, local_t, :]  # [B, Q, 1]
                all_tracks[:, :, global_t, :] = (
                    all_tracks[:, :, global_t, :] * weight_map[:, :, global_t, :]
                    + window_tracks[:, :, local_t, :] * w
                ) / (weight_map[:, :, global_t, :] + w).clamp_min(1e-6)
                all_vis[:, :, global_t] = (
                    all_vis[:, :, global_t] * weight_map[:, :, global_t, 0]
                    + vis[:, :, local_t] * w.squeeze(-1)
                ) / (weight_map[:, :, global_t, 0] + w.squeeze(-1)).clamp_min(1e-6)
                weight_map[:, :, global_t, :] += w

        return {
            "tracks": all_tracks,       # [B, Q, T_total, 2]
            "visibility": all_vis,      # [B, Q, T_total]
        }

    # ------------------------------------------------------------------
    # Training step helper
    # ------------------------------------------------------------------

    def training_step(
        self,
        frames: torch.Tensor,
        config,
        num_query_points: int = 64,
    ) -> dict:
        """Run a full self-supervised training step on a frame window.

        Samples random query points, tracks forward and backward, and
        computes all temporal self-supervised losses.

        Feature extraction happens once under ``torch.no_grad``;
        reversed-time features are obtained by flipping the cached tensors
        rather than re-encoding, keeping the cost at T backbone passes total.

        Args:
            frames: [B, T, 3, H, W] temporal window.
            config: DotMap config with loss weights.
            num_query_points: Number of query points to sample.

        Returns:
            dict with ``loss_total`` (differentiable) and ``metrics`` sub-dict.
        """
        from gatetracker.tracking.losses import compute_temporal_tracking_losses, _sample_query_grid

        B, T, _, H, W = frames.shape

        query_points = _sample_query_grid(B, H, W, num_query_points, frames.device)  # [B, Q, 2]

        # --- Feature extraction (single pass, frozen) ---
        descriptor_maps, fine_feature_maps = self._encode_all_frames(frames)

        if fine_feature_maps is None:
            raise RuntimeError(
                "TemporalTracker training requires fine feature maps. "
                "Enable REFINEMENT_METHOD=feature_softargmax in config."
            )

        feature_stride = getattr(self.matcher.model, "fine_feature_stride", 4)

        # --- Forward tracking ---
        coarse_tracks = self._global_coarse_match(query_points, descriptor_maps)
        query_fine = correspondence.sample_embeddings_at_points(
            fine_feature_maps[:, 0], query_points, feature_stride,
        )  # [B, Q, C_f]

        fwd_out = self.refinement_net(
            coarse_tracks, query_fine, fine_feature_maps, feature_stride,
        )
        tracks_fwd = fwd_out["tracks"]       # [B, Q, T, 2]
        vis_fwd = fwd_out["visibility"]      # [B, Q, T]

        # --- Backward tracking (reuse features, flip time) ---
        desc_maps_rev = descriptor_maps[::-1]
        fine_rev = fine_feature_maps.flip(dims=[1])

        rev_query = tracks_fwd[:, :, -1, :].detach()  # [B, Q, 2]
        coarse_rev = self._global_coarse_match(rev_query, desc_maps_rev)
        rev_query_fine = correspondence.sample_embeddings_at_points(
            fine_rev[:, 0], rev_query, feature_stride,
        )
        bwd_out = self.refinement_net(
            coarse_rev, rev_query_fine, fine_rev, feature_stride,
        )
        tracks_bwd = bwd_out["tracks"].flip(dims=[2])  # [B, Q, T, 2]

        # --- Losses ---
        patch_size = self.matcher.patch_size
        loss_dict = compute_temporal_tracking_losses(
            tracks=tracks_fwd,
            reverse_tracks=tracks_bwd,
            query_points=query_points,
            visibility=vis_fwd,
            descriptor_maps=descriptor_maps,
            fine_feature_maps=fine_feature_maps,
            patch_size=patch_size,
            feature_stride=feature_stride,
            config=config,
        )

        return loss_dict
