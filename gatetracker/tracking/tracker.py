"""
TemporalTracker — high-level wrapper combining a pretrained Matcher (frozen
descriptor backbone) with a TemporalRefinementNetwork for long-term point
tracking.

The tracker follows a two-stage TAPIR-style pipeline:
  Stage 1: Global coarse matching using pretrained descriptors (no new params).
  Stage 2: Iterative temporal refinement using local correlation + temporal
           convolutions (trainable TemporalRefinementNetwork).

Long-horizon design notes
-------------------------
``track_long_sequence`` anchors every sliding window to the **original** query
descriptor and the **original** fine feature sampled at frame 0 of the global
sequence. Without this anchor, each new window re-samples the descriptor at the
(already-drifting) tracked position — descriptor drift then accumulates linearly
with the number of windows, which is fatal beyond a few dozen frames. The
backbone is also encoded only once for the whole sequence and sliced per window,
so the cost of long inference is linear in T_total rather than ~2× redundant.
"""

from typing import Optional

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
        encoder_chunk_size: int = 16,
    ):
        super().__init__()
        self.matcher = matcher
        self.refinement_net = refinement_net
        self._freeze_matcher = freeze_matcher
        self.encoder_chunk_size = max(1, int(encoder_chunk_size))

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
            TEMPORAL_REFINEMENT_SOFTMAX_TEMPERATURE
                (falls back to FINE_REFINEMENT_TEMPERATURE, then 0.1)
            TRACKING_ENCODER_CHUNK_SIZE (16)
                Max frames per backbone forward pass in ``_encode_all_frames``.
                Increasing this raises GPU utilization during feature extraction.
            FINE_REFINEMENT_DIM (64)
        """
        feature_dim = int(config.get("FINE_REFINEMENT_DIM", 64))
        softmax_temperature = float(
            config.get(
                "TEMPORAL_REFINEMENT_SOFTMAX_TEMPERATURE",
                config.get("FINE_REFINEMENT_TEMPERATURE", 0.1),
            )
        )
        encoder_chunk_size = int(config.get("TRACKING_ENCODER_CHUNK_SIZE", 16))
        refinement_net = TemporalRefinementNetwork(
            feature_dim=feature_dim,
            hidden_dim=int(config.get("TEMPORAL_REFINEMENT_HIDDEN_DIM", 128)),
            num_blocks=int(config.get("TEMPORAL_REFINEMENT_NUM_BLOCKS", 4)),
            num_iters=int(config.get("TEMPORAL_REFINEMENT_NUM_ITERS", 4)),
            corr_radius=int(config.get("TEMPORAL_REFINEMENT_CORR_RADIUS", 3)),
            kernel_size=int(config.get("TEMPORAL_REFINEMENT_KERNEL_SIZE", 5)),
            softmax_temperature=softmax_temperature,
        ).to(matcher.device)

        return TemporalTracker(
            matcher,
            refinement_net,
            freeze_matcher=True,
            encoder_chunk_size=encoder_chunk_size,
        )

    # ------------------------------------------------------------------
    # Feature extraction helpers (frozen backbone)
    # ------------------------------------------------------------------

    def _encode_all_frames(self, frames: torch.Tensor, chunk_size: int | None = None):
        """Encode all frames in a window through the frozen backbone.

        Processes frames in chunks of ``chunk_size`` to avoid OOM while
        still being significantly faster than a per-frame loop.

        Args:
            frames:     [B, T, 3, H, W]
            chunk_size: Max images per backbone forward pass. When ``None``,
                falls back to ``self.encoder_chunk_size`` (set via the
                ``TRACKING_ENCODER_CHUNK_SIZE`` config key).

        Returns:
            descriptor_maps: list of T tensors [B, C, H_p, W_p]
            fine_feature_maps: [B, T, C_f, H_f, W_f] or None
        """
        if chunk_size is None:
            chunk_size = self.encoder_chunk_size
        chunk_size = max(1, int(chunk_size))

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
        query_descriptors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Match query descriptors against every frame independently.

        Args:
            query_points:      [B, Q, 2] positions at the **anchor** frame
                (used only when ``query_descriptors`` is None).
            descriptor_maps:   list of T tensors [B, C, H_p, W_p].
            query_descriptors: Optional [B, Q, C] L2-normalised query
                descriptors. When provided, they are used directly so the
                anchor descriptor stays consistent across long horizons
                (see ``track_long_sequence``). When ``None`` the descriptor
                is sampled from ``descriptor_maps[0]`` at ``query_points``
                (legacy single-window behaviour).

        Returns:
            coarse_tracks: [B, Q, T, 2] patch-center coordinates.
        """
        patch_size = self.matcher.patch_size
        T = len(descriptor_maps)

        if query_descriptors is None:
            query_desc = correspondence.sample_embeddings_at_points(
                descriptor_maps[0], query_points, patch_size,
            )  # [B, Q, C]
        else:
            query_desc = query_descriptors  # [B, Q, C]

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
        query_descriptors: Optional[torch.Tensor] = None,
        query_features: Optional[torch.Tensor] = None,
        descriptor_maps: Optional[list[torch.Tensor]] = None,
        fine_feature_maps: Optional[torch.Tensor] = None,
    ) -> dict:
        """Track query points across a temporal window.

        Args:
            query_points: [B, Q, 2] pixel coordinates at the anchor frame.
                Used only when the corresponding ``query_*`` argument is None
                (then descriptors / fine features are sampled at these
                positions from frame 0 of the window).
            frames:       [B, T, 3, H, W] frame window. Ignored when both
                ``descriptor_maps`` and ``fine_feature_maps`` are provided.
            query_descriptors: Optional [B, Q, C_desc] anchor descriptor
                used by the global coarse match. Set this to keep the
                semantic anchor fixed across long horizons.
            query_features:    Optional [B, Q, C_fine] anchor fine feature
                used by the temporal refinement correlation. Set this to
                keep the refinement anchor fixed across long horizons.
            descriptor_maps:   Optional precomputed list of T tensors
                [B, C, H_p, W_p]. When provided the backbone is not invoked.
            fine_feature_maps: Optional precomputed [B, T, C_f, H_f, W_f].
                When provided the backbone is not invoked.

        Returns:
            dict with:
                tracks:       [B, Q, T, 2] refined pixel coordinates.
                visibility:   [B, Q, T] visibility logits.
                coarse_tracks: [B, Q, T, 2] pre-refinement positions.
        """
        if descriptor_maps is None or fine_feature_maps is None:
            descriptor_maps, fine_feature_maps = self._encode_all_frames(frames)

        coarse_tracks = self._global_coarse_match(
            query_points, descriptor_maps, query_descriptors=query_descriptors,
        )  # [B, Q, T, 2]

        if fine_feature_maps is None:
            return {
                "tracks": coarse_tracks,
                "visibility": torch.ones(
                    coarse_tracks.shape[:3], device=coarse_tracks.device,
                ),
                "coarse_tracks": coarse_tracks,
            }

        feature_stride = getattr(self.matcher.model, "fine_feature_stride", 4)
        if query_features is None:
            query_features = correspondence.sample_embeddings_at_points(
                fine_feature_maps[:, 0], query_points, feature_stride,
            )  # [B, Q, C_f]

        refined = self.refinement_net(
            coarse_tracks=coarse_tracks,
            query_features=query_features,
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
        stride: Optional[int] = None,
        encoder_chunk_size: Optional[int] = None,
    ) -> dict:
        """Track points across a long sequence using sliding overlapping windows.

        Long-horizon design:

        - The backbone is encoded **once** for the whole sequence and the
          resulting per-frame descriptor / fine-feature maps are sliced per
          window. Per-window re-encoding is wasteful and was the dominant
          cost of the previous implementation.
        - The **original** query descriptor (coarse) and **original** fine
          feature (sampled at frame 0 of the global sequence at
          ``query_points``) are reused for every window. This prevents the
          slow descriptor drift that is otherwise unavoidable when later
          windows re-sample their anchor at the already-tracked position.
        - Per-frame estimates from overlapping windows are aggregated as a
          visibility-probability-weighted average in probability space; the
          final visibility is converted back to a logit so downstream
          consumers (loss / sigmoid > 0.5 thresholding) can keep the same
          interface.

        Args:
            query_points:        [B, Q, 2] initial positions at frame 0.
            frames:              [B, T_total, 3, H, W] full sequence.
            window_size:         Temporal window length per refinement call.
            stride:              Window step. Defaults to ``window_size // 2``.
            encoder_chunk_size:  Override for the per-call backbone chunk
                size. Defaults to ``self.encoder_chunk_size``.

        Returns:
            dict with tracks [B, Q, T_total, 2] and visibility [B, Q, T_total]
            (logits, same convention as ``track``).
        """
        B, T_total = frames.shape[:2]
        Q = query_points.shape[1]
        device = frames.device
        if stride is None or stride <= 0:
            stride = max(1, window_size // 2)
        stride = int(min(max(stride, 1), window_size))

        descriptor_maps_all, fine_feature_maps_all = self._encode_all_frames(
            frames, chunk_size=encoder_chunk_size,
        )

        patch_size = self.matcher.patch_size
        feature_stride = getattr(self.matcher.model, "fine_feature_stride", 4)

        query_desc_anchor = correspondence.sample_embeddings_at_points(
            descriptor_maps_all[0], query_points, patch_size,
        )  # [B, Q, C_desc]
        if fine_feature_maps_all is not None:
            query_fine_anchor = correspondence.sample_embeddings_at_points(
                fine_feature_maps_all[:, 0], query_points, feature_stride,
            )  # [B, Q, C_fine]
        else:
            query_fine_anchor = None

        all_tracks = torch.zeros(B, Q, T_total, 2, device=device)
        all_vis_prob = torch.zeros(B, Q, T_total, device=device)
        weight_map = torch.zeros(B, Q, T_total, device=device)

        all_tracks[:, :, 0, :] = query_points
        all_vis_prob[:, :, 0] = 1.0
        weight_map[:, :, 0] = 1.0

        for w_start in range(0, T_total - 1, stride):
            w_end = min(w_start + window_size, T_total)
            if w_end - w_start < 2:
                break

            window_descs = descriptor_maps_all[w_start:w_end]
            window_fine = (
                fine_feature_maps_all[:, w_start:w_end]
                if fine_feature_maps_all is not None
                else None
            )

            with torch.no_grad():
                out = self.track(
                    query_points=query_points,
                    frames=frames[:, w_start:w_end],  # ignored (caches provided)
                    query_descriptors=query_desc_anchor,
                    query_features=query_fine_anchor,
                    descriptor_maps=window_descs,
                    fine_feature_maps=window_fine,
                )

            window_tracks = out["tracks"]  # [B, Q, T_w, 2]
            window_vis = out["visibility"]  # [B, Q, T_w] logits
            window_prob = torch.sigmoid(window_vis)  # [B, Q, T_w]

            local_idx = torch.arange(w_end - w_start, device=device)
            global_idx = w_start + local_idx  # [T_w]

            cur_w = weight_map[:, :, global_idx]  # [B, Q, T_w]
            new_w = window_prob  # [B, Q, T_w]
            denom = (cur_w + new_w).clamp_min(1e-6)

            all_tracks[:, :, global_idx, :] = (
                all_tracks[:, :, global_idx, :] * cur_w.unsqueeze(-1)
                + window_tracks * new_w.unsqueeze(-1)
            ) / denom.unsqueeze(-1)
            all_vis_prob[:, :, global_idx] = (
                all_vis_prob[:, :, global_idx] * cur_w
                + window_prob * new_w
            ) / denom
            weight_map[:, :, global_idx] = cur_w + new_w

        # Anchor the global frame 0 exactly at the user-provided query
        # locations: the refinement net's soft-argmax can otherwise drift the
        # anchor by sub-pixel amounts and that is visually distracting in the
        # rendered tracks video.
        all_tracks[:, :, 0, :] = query_points

        # Convert aggregated probabilities back to logits for downstream
        # consumers that expect ``visibility`` in logit space.
        eps = 1e-6
        p = all_vis_prob.clamp(eps, 1.0 - eps)
        all_vis_logits = torch.log(p / (1.0 - p))

        return {
            "tracks": all_tracks,         # [B, Q, T_total, 2]
            "visibility": all_vis_logits,  # [B, Q, T_total] logits
        }

    # ------------------------------------------------------------------
    # Training step helper
    # ------------------------------------------------------------------

    def training_step(
        self,
        frames: torch.Tensor,
        config,
        num_query_points: int = 64,
        batch=None,
        geometry_pipeline=None,
        epoch: int = 0,
    ) -> dict:
        """Run a temporal training step (self-supervised ± pseudo-GT supervision).

        When ``PSEUDO_GT_MIX`` / ``PSEUDO_GT_SUP_LAMBDA_MAX`` trigger and
        ``geometry_pipeline`` is set, replaces the window with on-the-fly
        novel-view pseudo sequences and adds masked supervised losses.

        Args:
            frames: [B, T, 3, H, W] temporal window.
            config: DotMap config with loss weights.
            num_query_points: Target number of query points (subsampled pseudo grid).
            batch: Optional dataloader batch (reserved for future keys).
            geometry_pipeline: :class:`~gatetracker.geometry.pipeline.GeometryPipeline`.
            epoch: Current epoch (for curriculum on supervised weight).

        Returns:
            dict with ``loss_total`` (differentiable) and ``metrics`` sub-dict.
        """
        from gatetracker.data.pseudo_gt import (
            DeformationConfig,
            GridConfig,
            PseudoGTGenerator,
            TrajectoryConfig,
        )
        from gatetracker.tracking.losses import (
            _sample_query_grid,
            composite_supervision_mask,
            compute_temporal_tracking_losses,
            temporal_supervised_losses,
            validity_at_tracks_bqt,
        )

        B, T, _, H, W = frames.shape
        device = frames.device

        lam_max = float(config.get("PSEUDO_GT_SUP_LAMBDA_MAX", 0.0))
        mix = float(config.get("PSEUDO_GT_MIX", 0.0))
        use_pseudo = (
            lam_max > 0.0
            and mix > 0.0
            and geometry_pipeline is not None
            and torch.rand((), device=device).item() < mix
        )
        _ = batch  # reserved for offline GT / paths

        frames_in = frames
        query_points = _sample_query_grid(B, H, W, num_query_points, device)  # [B, Q, 2]
        cycle_weight_scale = 1.0
        synth_self_sup_scale = float(config.get("PSEUDO_GT_SYNTH_SELF_SUP_SCALE", 1.0))
        self_sup_mask_bqt = None
        tracks_gt = None
        vis_gt = None
        vis_target = None
        composite_m = None

        if use_pseudo:
            key = (H, W, str(device))
            if getattr(self, "_pseudo_gt_hw_device", None) != key:
                self._pseudo_gt_gen = PseudoGTGenerator(H, W, device=str(device))
                self._pseudo_gt_hw_device = key

            grid_sz = int(config.get("PSEUDO_GT_GRID_SIZE", 32))
            grid_cfg = GridConfig(
                grid_size=grid_sz,
                margin_frac=float(config.get("PSEUDO_GT_GRID_MARGIN_FRAC", 0.03)),
            )
            erode = int(config.get("PSEUDO_GT_MASK_ERODE_PX", 0))

            with torch.no_grad():
                depth, _, K = geometry_pipeline.compute_geometry(
                    frames[:, 0], return_normalized=False,
                )

            synth_list = []
            tg_list = []
            vg_list = []
            fv_list = []
            qp_list = []
            Q_full = grid_sz * grid_sz

            for b in range(B):
                seed_u = int(epoch * 100003 + b * 17 + 1)
                res = self._pseudo_gt_gen.generate(
                    image=frames[b : b + 1, 0],
                    depth=depth[b : b + 1],
                    intrinsics=K[b : b + 1],
                    trajectory=TrajectoryConfig(n_frames=T),
                    deformation=DeformationConfig(),
                    grid=grid_cfg,
                    seed=seed_u,
                    frame_valid_erode_px=erode,
                )
                synth_list.append(res.frames)  # [T, 3, H, W]
                tr = res.tracks  # [T, Q_full, 2]
                vi = res.visibility.to(dtype=torch.float32, device=device)
                fv = res.frame_valid  # [T, 1, H, W]
                qp = res.query_pixels  # [Q_full, 2]

                Qeff = min(num_query_points, Q_full)
                g = torch.Generator(device=device)
                g.manual_seed(seed_u)
                idx = torch.randperm(Q_full, generator=g, device=device)[:Qeff]

                tr_s = tr[:, idx, :]  # [T, Qeff, 2]
                vi_s = vi[:, idx]
                qp_s = qp[idx, :]

                tg_list.append(tr_s.permute(1, 0, 2).unsqueeze(0))  # [1, Qeff, T, 2]
                vg_list.append(vi_s.permute(1, 0).unsqueeze(0))  # [1, Qeff, T]
                fv_list.append(fv.unsqueeze(0))  # [1, T, 1, H, W]
                qp_list.append(qp_s.unsqueeze(0))

            frames_in = torch.stack(synth_list, dim=0)  # [B, T, 3, H, W]
            tracks_gt = torch.cat(tg_list, dim=0)  # [B, Q, T, 2]
            vis_gt = torch.cat(vg_list, dim=0)  # [B, Q, T]
            frame_valid_bt = torch.cat(fv_list, dim=0)  # [B, T, 1, H, W]
            query_points = torch.cat(qp_list, dim=0)  # [B, Q, 2]

            w_rgb = validity_at_tracks_bqt(frame_valid_bt, tracks_gt, H, W)
            composite_m = composite_supervision_mask(
                vis_gt, w_rgb, tracks_gt, H, W,
            )
            self_sup_mask_bqt = composite_m

            appearance_aware = bool(config.get("PSEUDO_GT_VIS_APPEARANCE_AWARE", False))
            if appearance_aware:
                vis_target = vis_gt * w_rgb
            else:
                vis_target = vis_gt

            cycle_weight_scale = 0.0

        # --- Feature extraction (single pass, frozen) ---
        descriptor_maps, fine_feature_maps = self._encode_all_frames(frames_in)

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
        if cycle_weight_scale > 0.0:
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
        else:
            tracks_bwd = tracks_fwd.detach()

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
            self_sup_mask_bqt=self_sup_mask_bqt,
            cycle_weight_scale=cycle_weight_scale,
            synth_self_sup_scale=synth_self_sup_scale,
        )

        loss_total = loss_dict["loss_total"]

        if use_pseudo and tracks_gt is not None and vis_target is not None and composite_m is not None:
            sup = temporal_supervised_losses(
                tracks_pred=tracks_fwd,
                visibility_pred=vis_fwd,
                tracks_gt=tracks_gt,
                composite_mask=composite_m,
                vis_target=vis_target,
                config=config,
            )
            cur_epochs = max(1, int(config.get("PSEUDO_GT_CURRICULUM_EPOCHS", 10)))
            lam = lam_max * min(1.0, float(epoch + 1) / float(cur_epochs))
            loss_total = (1.0 - lam) * loss_total + lam * sup["loss_sup_total"]
            loss_dict["metrics"].update(sup["metrics"])
            loss_dict["metrics"]["pseudo_lambda"] = lam
            loss_dict["metrics"]["pseudo_gt_active"] = 1.0
        else:
            loss_dict["metrics"]["pseudo_gt_active"] = 0.0
            loss_dict["metrics"]["pseudo_lambda"] = 0.0

        loss_dict["loss_total"] = loss_total
        loss_dict["metrics"]["loss_total"] = float(loss_total.detach().item())

        return loss_dict
