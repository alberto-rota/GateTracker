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
import warnings

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from gatetracker.distributed_context import unwrap_model
from gatetracker.matching import correspondence
from gatetracker.matching.matcher import Matcher
from gatetracker.tracking.temporal_refinement import TemporalRefinementNetwork
from gatetracker.utils.tensor_ops import embedding2chw, chw2embedding

# ``torch.utils.checkpoint`` wraps recomputation in ``torch.cpu.amp.autocast``,
# which is deprecated. That path runs during **backward**, so a local
# ``catch_warnings`` around ``checkpoint()`` does not cover it — register once.
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch\.cpu\.amp\.autocast.*",
)


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
            TEMPORAL_POINT_ATTN (False)            — enable cross-point attention.
            TEMPORAL_POINT_ATTN_HEADS (4)
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
            use_point_attention=bool(config.get("TEMPORAL_POINT_ATTN", False)),
            point_attention_heads=int(config.get("TEMPORAL_POINT_ATTN_HEADS", 4)),
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
        # DDP does not expose private methods like `_encode_image` on the wrapper.
        backbone = unwrap_model(self.matcher.model)

        with torch.no_grad():
            for start in range(0, N_total, chunk_size):
                chunk = flat_frames[start : start + chunk_size]  # [C_bs, 3, H, W]
                _, matched_tokens, _, _, fine_map, _ = backbone._encode_image(chunk)
                emb = matched_tokens.permute(0, 2, 1)  # [C_bs, C, N]
                if backbone.resampled_patch_size != 16:
                    emb = embedding2chw(emb, embed_dim_last=False)
                    emb = backbone.patchsize_resampler(emb)
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

    @staticmethod
    def _coarse_match_similarity(
        query_desc_bqkc: torch.Tensor,
        target_flat_bcn: torch.Tensor,
    ) -> torch.Tensor:
        """Similarity of each query's anchor bank to every target patch.

        Args:
            query_desc_bqkc:  ``[B, Q, K, C]`` L2-normalised query bank.
            target_flat_bcn:  ``[B, C, N]`` target patches (will be re-normalised).

        Returns:
            ``[B, Q, N]`` max-over-anchors cosine similarity.
        """
        B, Q, K, C = query_desc_bqkc.shape
        target_norm = torch.nn.functional.normalize(target_flat_bcn, dim=1)  # [B, C, N]
        # Fold K into Q for a single bmm, then reduce.
        qd = query_desc_bqkc.reshape(B, Q * K, C)
        sim_qk = torch.bmm(qd, target_norm)  # [B, Q*K, N]
        sim = sim_qk.reshape(B, Q, K, -1).amax(dim=2)  # [B, Q, N]
        return sim

    def _global_coarse_match(
        self,
        query_points: torch.Tensor,
        descriptor_maps: list[torch.Tensor],
        query_descriptors: Optional[torch.Tensor] = None,
        prev_positions: Optional[torch.Tensor] = None,
        redetect_mask: Optional[torch.Tensor] = None,
        redetect_topk: int = 1,
    ) -> torch.Tensor:
        """Match query descriptors against every frame independently.

        Args:
            query_points:      [B, Q, 2] positions at the **anchor** frame
                (used only when ``query_descriptors`` is None).
            descriptor_maps:   list of T tensors [B, C, H_p, W_p].
            query_descriptors: Optional [B, Q, C] or [B, Q, K, C] L2-normalised
                query descriptors. When a bank axis ``K`` is present, the
                per-patch similarity is the ``max`` over anchors (single-anchor
                behaviour recovered at ``K = 1``). When ``None`` the descriptor
                is sampled from ``descriptor_maps[0]`` at ``query_points``
                (legacy single-window behaviour).
            prev_positions:    Optional ``[B, Q, 2]`` previous tracked position
                per query (e.g. last visible frame). When provided together
                with ``redetect_mask``, the coarse match for re-detected
                queries is replaced by a refinement-score re-ranked top-K
                candidate (see below). When only ``prev_positions`` is given
                this argument is ignored (current behaviour).
            redetect_mask:     Optional ``[B, Q]`` bool flag per query marking
                entries that have been occluded for a long run and should be
                re-detected by re-ranking top-K coarse candidates per frame.
                Requires ``redetect_topk > 1``.
            redetect_topk:     Candidates per frame considered for re-detection.
                ``1`` disables re-ranking (pure argmax). The re-ranker picks
                the candidate whose local correlation peak is highest; this
                additional lookup is only triggered for the flagged queries.

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
            query_desc = query_descriptors

        # Always work in [B, Q, K, C] form internally.
        if query_desc.dim() == 3:
            query_desc = query_desc.unsqueeze(2)  # [B, Q, 1, C]

        # ``redetect_mask`` requires at least 2 candidates to be meaningful.
        topk = max(1, int(redetect_topk))
        use_redetect = (
            redetect_mask is not None and topk > 1 and redetect_mask.any()
        )

        coarse_positions = []
        for t in range(T):
            target_emb = descriptor_maps[t]
            B, C, H_p, W_p = target_emb.shape
            N = H_p * W_p
            target_flat = target_emb.reshape(B, C, N)  # [B, C, N]
            sim = self._coarse_match_similarity(query_desc, target_flat)  # [B, Q, N]

            if use_redetect and topk > 1:
                # Top-K candidates per (b, q) frame; shape [B, Q, topk].
                topk_vals, topk_idx = sim.topk(min(topk, N), dim=-1)
                # Score candidates: base is cosine similarity (already computed).
                # When no sophisticated re-ranker is required we simply keep
                # the argmax — we still return top-K so callers can mix in a
                # locality prior if ``prev_positions`` is given.
                if prev_positions is not None:
                    half_patch = patch_size // 2
                    # Candidate pixel coords for every (b, q, k).
                    tgt_y = (topk_idx // W_p).float()
                    tgt_x = (topk_idx % W_p).float()
                    cand_x = tgt_x * patch_size + half_patch  # [B, Q, topk]
                    cand_y = tgt_y * patch_size + half_patch  # [B, Q, topk]
                    cand_xy = torch.stack([cand_x, cand_y], dim=-1)  # [B, Q, topk, 2]
                    dxy = cand_xy - prev_positions.unsqueeze(2)
                    dist = dxy.norm(dim=-1)  # [B, Q, topk]
                    # Soft locality prior (sigma = image size / 4) added to sim.
                    img_diag = float((H_p * patch_size) ** 2 + (W_p * patch_size) ** 2) ** 0.5
                    sigma = max(1.0, img_diag / 6.0)
                    locality = torch.exp(-(dist / sigma) ** 2)  # [B, Q, topk]
                    score = topk_vals + 0.25 * locality
                    best_k = score.argmax(dim=-1, keepdim=True)  # [B, Q, 1]
                    best_idx = topk_idx.gather(-1, best_k).squeeze(-1)  # [B, Q]
                else:
                    best_idx = topk_idx[..., 0]
                # For queries NOT flagged for redetect, keep pure argmax.
                rmask = redetect_mask.to(dtype=torch.bool)  # [B, Q]
                argmax_idx = sim.argmax(dim=-1)  # [B, Q]
                best_idx = torch.where(rmask, best_idx, argmax_idx)
            else:
                best_idx = sim.argmax(dim=-1)  # [B, Q]

            half_patch = patch_size // 2
            tgt_y = best_idx // W_p
            tgt_x = best_idx % W_p
            tgt_px_x = (tgt_x * patch_size + half_patch).float()
            tgt_px_y = (tgt_y * patch_size + half_patch).float()
            coarse_pos = torch.stack([tgt_px_x, tgt_px_y], dim=-1)  # [B, Q, 2]
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
        num_iters: Optional[int] = None,
        prev_positions: Optional[torch.Tensor] = None,
        redetect_mask: Optional[torch.Tensor] = None,
        redetect_topk: int = 1,
    ) -> dict:
        """Track query points across a temporal window.

        Args:
            query_points: [B, Q, 2] pixel coordinates at the anchor frame.
                Used only when the corresponding ``query_*`` argument is None
                (then descriptors / fine features are sampled at these
                positions from frame 0 of the window).
            frames:       [B, T, 3, H, W] frame window. Ignored when both
                ``descriptor_maps`` and ``fine_feature_maps`` are provided.
            query_descriptors: Optional [B, Q, C_desc] or [B, Q, K, C_desc]
                anchor descriptor(s) used by the global coarse match. With an
                anchor bank of size ``K`` the coarse match takes the max over
                anchors. Set this to keep the semantic anchor fixed across
                long horizons.
            query_features:    Optional [B, Q, C_fine] or [B, Q, K, C_fine]
                anchor fine feature(s) used by the temporal refinement
                correlation. Bank reduction is also via ``max``.
            descriptor_maps:   Optional precomputed list of T tensors
                [B, C, H_p, W_p]. When provided the backbone is not invoked.
            fine_feature_maps: Optional precomputed [B, T, C_f, H_f, W_f].
                When provided the backbone is not invoked.
            num_iters:         Optional override on the number of refinement
                iterations (e.g. more iters at inference than train).
            prev_positions:    Optional ``[B, Q, 2]`` previous tracked position
                per query, used by the top-K re-detection re-ranker to prefer
                spatially plausible candidates.
            redetect_mask:     Optional ``[B, Q]`` bool flag marking queries
                that should be re-detected (top-K + re-ranking) rather than
                pure argmax-coarse-matched.
            redetect_topk:     Candidates per frame when ``redetect_mask`` is
                set. ``1`` reproduces the legacy argmax behaviour.

        Returns:
            dict with:
                tracks:       [B, Q, T, 2] refined pixel coordinates.
                visibility:   [B, Q, T] visibility logits.
                coarse_tracks: [B, Q, T, 2] pre-refinement positions.
        """
        if descriptor_maps is None or fine_feature_maps is None:
            descriptor_maps, fine_feature_maps = self._encode_all_frames(frames)

        coarse_tracks = self._global_coarse_match(
            query_points, descriptor_maps,
            query_descriptors=query_descriptors,
            prev_positions=prev_positions,
            redetect_mask=redetect_mask,
            redetect_topk=redetect_topk,
        )  # [B, Q, T, 2]

        if fine_feature_maps is None:
            return {
                "tracks": coarse_tracks,
                "visibility": torch.ones(
                    coarse_tracks.shape[:3], device=coarse_tracks.device,
                ),
                "coarse_tracks": coarse_tracks,
            }

        feature_stride = getattr(
            unwrap_model(self.matcher.model), "fine_feature_stride", 4,
        )
        if query_features is None:
            query_features = correspondence.sample_embeddings_at_points(
                fine_feature_maps[:, 0], query_points, feature_stride,
            )  # [B, Q, C_f]

        refined = self.refinement_net(
            coarse_tracks=coarse_tracks,
            query_features=query_features,
            feature_maps=fine_feature_maps,
            feature_stride=feature_stride,
            num_iters=num_iters,
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
        vis_agg: str = "mean",
        infer_max_step_px: float = 0.0,
        num_iters: Optional[int] = None,
        centrality_weighting: bool = False,
        anchor_bank_size: int = 1,
        anchor_refresh_vis_thresh: float = 0.9,
        redetection: bool = False,
        redetect_after: int = 5,
        redetect_topk: int = 8,
        redetect_vis_thresh: float = 0.3,
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
            vis_agg:             ``mean`` (weighted average) or ``min`` (pessimistic
                fusion across overlapping windows) for visibility probability.
            infer_max_step_px:   If > 0, clamp each frame-to-frame displacement
                to this many pixels after fusion (0 = disabled).
            num_iters:           Optional override for the refinement iteration
                count at inference. Weights are shared across iters.
            centrality_weighting: If ``True``, down-weight each window's
                contribution at its temporal edges. The weight is a triangular
                function of local frame index that peaks at the window centre,
                modelling the fact that the temporal mixer has more context at
                the middle than at the start/end.
            anchor_bank_size:    Bank of template descriptors kept per query.
                ``1`` reproduces the legacy single-anchor behaviour. With
                ``K > 1``, frame 0 is always kept and the remaining ``K - 1``
                slots hold the most recently refreshed high-confidence
                appearances (see ``anchor_refresh_vis_thresh``).
            anchor_refresh_vis_thresh: Predicted visibility probability a frame
                must exceed to be considered for anchor refresh. Refreshes only
                happen for queries whose bank is not yet full or whose new
                slot would improve anchor temporal diversity.
            redetection:         If ``True``, re-detect queries whose fused
                visibility has been below ``redetect_vis_thresh`` for more
                than ``redetect_after`` frames. Re-detection uses a top-K
                coarse search plus locality re-ranking against the last
                trusted position (anchor frame for that query).
            redetect_after:      Minimum run-length of low visibility before
                re-detection triggers.
            redetect_topk:       Candidates evaluated during re-detection.
            redetect_vis_thresh: Visibility probability below which a frame is
                considered occluded for the re-detection counter.

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
        feature_stride = getattr(
            unwrap_model(self.matcher.model), "fine_feature_stride", 4,
        )

        # --- Template bank (multi-anchor) -----------------------------------
        K_bank = max(1, int(anchor_bank_size))
        query_desc_0 = correspondence.sample_embeddings_at_points(
            descriptor_maps_all[0], query_points, patch_size,
        )  # [B, Q, C_desc]
        # Bank buffers hold ``K_bank`` anchors per query; slot 0 is the pinned
        # frame-0 anchor. Slot index 1..K_bank-1 are filled lazily.
        desc_bank = query_desc_0.unsqueeze(2).expand(-1, -1, K_bank, -1).clone()
        bank_filled = torch.zeros(B, Q, K_bank, device=device, dtype=torch.bool)
        bank_filled[:, :, 0] = True

        if fine_feature_maps_all is not None:
            query_fine_0 = correspondence.sample_embeddings_at_points(
                fine_feature_maps_all[:, 0], query_points, feature_stride,
            )  # [B, Q, C_fine]
            fine_bank = query_fine_0.unsqueeze(2).expand(-1, -1, K_bank, -1).clone()
        else:
            fine_bank = None

        all_tracks = torch.zeros(B, Q, T_total, 2, device=device)
        all_vis_prob = torch.zeros(B, Q, T_total, device=device)
        weight_map = torch.zeros(B, Q, T_total, device=device)
        vis_mode = str(vis_agg).lower().strip()
        use_vis_min = vis_mode in ("min", "pessimistic")

        all_tracks[:, :, 0, :] = query_points
        all_vis_prob[:, :, 0] = 1.0
        weight_map[:, :, 0] = 1.0
        if use_vis_min:
            all_vis_prob[:, :, 1:] = 1.0

        # Low-vis run counter for re-detection.
        low_vis_run = torch.zeros(B, Q, device=device, dtype=torch.long)
        # Track the last "trusted" position per query (last frame where the
        # fused visibility was above ``redetect_vis_thresh``). Initialise to
        # the query point so locality re-ranking has a sensible prior.
        last_trusted_pos = query_points.clone()

        # Next bank slot to fill per (b, q). Start at 1 since slot 0 is frame 0.
        bank_next_slot = torch.ones(B, Q, device=device, dtype=torch.long)

        for w_start in range(0, T_total - 1, stride):
            w_end = min(w_start + window_size, T_total)
            T_w = w_end - w_start
            if T_w < 2:
                break

            window_descs = descriptor_maps_all[w_start:w_end]
            window_fine = (
                fine_feature_maps_all[:, w_start:w_end]
                if fine_feature_maps_all is not None
                else None
            )

            # Build redetect mask for this window (any query with a long
            # low-vis run OR whose last trusted position is stale).
            if redetection and w_start > 0:
                rmask = low_vis_run >= int(max(1, redetect_after))  # [B, Q]
                rtopk = int(max(1, redetect_topk))
            else:
                rmask = None
                rtopk = 1

            with torch.no_grad():
                out = self.track(
                    query_points=query_points,
                    frames=frames[:, w_start:w_end],  # ignored (caches provided)
                    query_descriptors=desc_bank,
                    query_features=fine_bank,
                    descriptor_maps=window_descs,
                    fine_feature_maps=window_fine,
                    num_iters=num_iters,
                    prev_positions=last_trusted_pos,
                    redetect_mask=rmask,
                    redetect_topk=rtopk,
                )

            window_tracks = out["tracks"]  # [B, Q, T_w, 2]
            window_vis = out["visibility"]  # [B, Q, T_w] logits
            window_prob = torch.sigmoid(window_vis)  # [B, Q, T_w]

            local_idx = torch.arange(T_w, device=device)
            global_idx = w_start + local_idx  # [T_w]

            # Centrality weight (triangular, peaks at window centre).
            if centrality_weighting and T_w > 2:
                centre = (T_w - 1) / 2.0
                ct = 1.0 - (local_idx.float() - centre).abs() / max(centre, 1.0)
                ct = ct.clamp_min(0.25)  # never fully zero out an edge frame
                centrality = ct.view(1, 1, T_w)  # [1, 1, T_w]
            else:
                centrality = torch.ones(1, 1, T_w, device=device)

            cur_w = weight_map[:, :, global_idx]  # [B, Q, T_w]
            new_w = window_prob * centrality       # [B, Q, T_w]
            denom = (cur_w + new_w).clamp_min(1e-6)

            all_tracks[:, :, global_idx, :] = (
                all_tracks[:, :, global_idx, :] * cur_w.unsqueeze(-1)
                + window_tracks * new_w.unsqueeze(-1)
            ) / denom.unsqueeze(-1)
            if use_vis_min:
                all_vis_prob[:, :, global_idx] = torch.minimum(
                    all_vis_prob[:, :, global_idx], window_prob,
                )
            else:
                all_vis_prob[:, :, global_idx] = (
                    all_vis_prob[:, :, global_idx] * cur_w
                    + window_prob * new_w
                ) / denom
            weight_map[:, :, global_idx] = cur_w + new_w

            # --- Update bank from this window's high-confidence frames ------
            if K_bank > 1:
                # Pick the single best-visibility frame in the window (excluding
                # frame 0 of the global sequence) above the refresh threshold.
                # This is cheap: one argmax over T_w per (b, q).
                local_vis = window_prob.clone()
                if w_start == 0:
                    local_vis[:, :, 0] = -1.0  # skip frame 0 (already pinned).
                best_local = local_vis.argmax(dim=-1)  # [B, Q]
                best_vis = local_vis.gather(-1, best_local.unsqueeze(-1)).squeeze(-1)
                refresh_mask = best_vis >= float(anchor_refresh_vis_thresh)  # [B, Q]

                if refresh_mask.any():
                    best_global = w_start + best_local  # [B, Q]
                    best_xy = window_tracks.gather(
                        2, best_local.view(B, Q, 1, 1).expand(B, Q, 1, 2),
                    ).squeeze(2)  # [B, Q, 2]

                    # Sample descriptor + fine feature at the refreshed pose.
                    # descriptor_maps_all is a python list → build a single
                    # [B, C, H_p, W_p] by gathering per-batch-item maps at the
                    # same frame index. For efficiency we assume a single
                    # frame per refresh round (rare frames get picked).
                    # Gather per (b) the descriptor map at best_global[b, :]:
                    # in practice most queries pick the same frame, so we
                    # just loop over the (small) set of unique global frames.
                    unique_frames, inv = torch.unique(best_global, return_inverse=True)
                    for uf_i, uf in enumerate(unique_frames.tolist()):
                        sel_bq = (inv == uf_i) & refresh_mask  # [B, Q]
                        if not sel_bq.any():
                            continue
                        # Per-batch selection mask.
                        for b in range(B):
                            bsel = sel_bq[b]
                            if not bsel.any():
                                continue
                            xy_b = best_xy[b : b + 1, bsel]  # [1, Qs, 2]
                            desc_map_b = descriptor_maps_all[uf][b : b + 1]  # [1, C, H_p, W_p]
                            new_desc = correspondence.sample_embeddings_at_points(
                                desc_map_b, xy_b, patch_size,
                            )  # [1, Qs, C]
                            # Slot to write into = bank_next_slot (wraps modulo K_bank-1 + 1).
                            slots_b = bank_next_slot[b, bsel]  # [Qs]
                            q_indices = torch.nonzero(bsel, as_tuple=False).squeeze(-1)
                            desc_bank[b, q_indices, slots_b, :] = new_desc[0]
                            bank_filled[b, q_indices, slots_b] = True

                            if fine_bank is not None and fine_feature_maps_all is not None:
                                fine_map_b = fine_feature_maps_all[b : b + 1, uf]
                                new_fine = correspondence.sample_embeddings_at_points(
                                    fine_map_b, xy_b, feature_stride,
                                )  # [1, Qs, C_fine]
                                fine_bank[b, q_indices, slots_b, :] = new_fine[0]

                            # Advance round-robin pointer in [1, K_bank-1].
                            next_slots = slots_b + 1
                            wrap = next_slots >= K_bank
                            next_slots[wrap] = 1
                            bank_next_slot[b, q_indices] = next_slots

            # --- Low-vis counter + last trusted position -------------------
            # Use this window's fused prob on global_idx for the counter
            # update. End-of-window state is all we need for the next window.
            fused_vis_tail = all_vis_prob[:, :, global_idx[-1]]  # [B, Q]
            trusted = fused_vis_tail >= float(redetect_vis_thresh)
            last_trusted_pos = torch.where(
                trusted.unsqueeze(-1),
                all_tracks[:, :, global_idx[-1], :],
                last_trusted_pos,
            )
            low_vis_run = torch.where(
                trusted,
                torch.zeros_like(low_vis_run),
                low_vis_run + 1,
            )

        # Anchor the global frame 0 exactly at the user-provided query
        # locations: the refinement net's soft-argmax can otherwise drift the
        # anchor by sub-pixel amounts and that is visually distracting in the
        # rendered tracks video.
        all_tracks[:, :, 0, :] = query_points

        max_step = float(infer_max_step_px)
        if max_step > 0.0 and T_total > 1:
            d = all_tracks[:, :, 1:, :] - all_tracks[:, :, :-1, :]  # [B, Q, T-1, 2]
            n = d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            scale = (max_step / n).clamp(max=1.0)
            all_tracks[:, :, 1:, :] = all_tracks[:, :, :-1, :] + d * scale

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
            dict with:

            - ``loss_total``: scalar tensor optimized (self-sup ± pseudo-GT blend).
            - ``metrics_self_sup``: detached self-supervised term scalars plus
              ``loss_self_sup_total`` (pre-blend self-sup objective).
            - ``metrics_pseudo_gt``: pseudo-GT supervised scalars (zeros when inactive),
              including ``pseudo_gt_active`` and ``pseudo_lambda``.
        """
        from gatetracker.data.pseudo_gt import (
            GridConfig,
            PseudoGTGenerator,
            deformation_config_from_run_config,
            occluder_config_from_run_config,
            trajectory_config_from_run_config,
        )
        from gatetracker.tracking.losses import (
            _sample_query_grid,
            composite_supervision_mask,
            compute_temporal_tracking_losses,
            position_supervision_mask,
            temporal_supervised_losses,
            validity_at_tracks_bqt,
        )

        B, T_win, _, H, W = frames.shape
        device = frames.device

        lam_max = float(config.get("PSEUDO_GT_SUP_LAMBDA_MAX", 0.0))
        # Per-epoch mix from engine (``_SCHED_PSEUDO_GT_MIX``); falls back to static PSEUDO_GT_MIX.
        mix = float(
            config.get("_SCHED_PSEUDO_GT_MIX", config.get("PSEUDO_GT_MIX", 0.0))
        )
        use_pseudo = (
            lam_max > 0.0
            and mix > 0.0
            and geometry_pipeline is not None
            and torch.rand((), device=device).item() < mix
        )
        _ = batch  # reserved for offline GT / paths

        # ``LONG`` mode: with probability ``TRACKING_LONGSEQ_TRAIN_PROB`` stretch
        # the pseudo-GT clip to ``PSEUDO_GT_LONG_LENGTH`` frames and train the
        # refinement on chained sub-windows of size ``TRACKING_SEQUENCE_LENGTH``.
        # Only active on the pseudo-GT path (the only one that produces a GT
        # long trajectory). The self-sup path stays at ``T_win``.
        longseq_prob = float(config.get("TRACKING_LONGSEQ_TRAIN_PROB", 0.0))
        long_L_cfg = int(config.get("PSEUDO_GT_LONG_LENGTH", T_win))
        use_long = (
            use_pseudo
            and longseq_prob > 0.0
            and long_L_cfg > T_win
            and torch.rand((), device=device).item() < longseq_prob
        )
        L = max(T_win, long_L_cfg) if use_long else T_win
        T = L  # effective temporal length used by the downstream path

        frames_in = frames
        query_points = _sample_query_grid(B, H, W, num_query_points, device)  # [B, Q, 2]
        cycle_weight_scale = 1.0
        synth_self_sup_scale = float(config.get("PSEUDO_GT_SYNTH_SELF_SUP_SCALE", 1.0))
        self_sup_mask_bqt = None
        tracks_gt = None
        vis_gt = None
        vis_target = None
        composite_m = None
        pos_mask_m = None

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
            traj_cfg = trajectory_config_from_run_config(config, n_frames=L)
            deform_cfg = deformation_config_from_run_config(config)
            occ_cfg = occluder_config_from_run_config(config)

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
                    trajectory=traj_cfg,
                    deformation=deform_cfg,
                    grid=grid_cfg,
                    occluders=occ_cfg,
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
            # Broader mask that keeps occluded-but-in-bounds frames so the
            # position loss can teach the tracker to predict the pre-occlusion
            # pixel coordinate through short occlusions.
            pos_mask_m = position_supervision_mask(w_rgb, tracks_gt, H, W)
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

        feature_stride = getattr(
            unwrap_model(self.matcher.model), "fine_feature_stride", 4,
        )

        # --- Anchor bank assembly (train-time) --------------------------------
        # With a bank of size ``K_bank > 1`` we train the refinement to use
        # several templates simultaneously (max over anchors in correlation).
        # For the pseudo-GT path we can sample extra anchor frames at GT-visible
        # positions; for the pure self-sup path we fall back to K=1 (frame 0
        # only) since we have no trustworthy alternate anchor.
        K_bank = max(1, int(config.get("TRACKING_ANCHOR_BANK_SIZE", 1)))
        patch_size = self.matcher.patch_size

        query_desc_0 = correspondence.sample_embeddings_at_points(
            descriptor_maps[0], query_points, patch_size,
        )  # [B, Q, C_desc]
        query_fine_0 = correspondence.sample_embeddings_at_points(
            fine_feature_maps[:, 0], query_points, feature_stride,
        )  # [B, Q, C_f]

        if K_bank > 1 and use_pseudo and tracks_gt is not None and vis_gt is not None:
            desc_bank_list = [query_desc_0]
            fine_bank_list = [query_fine_0]
            # Sample K_bank - 1 additional anchor frames (shared across the
            # batch for efficiency). Per-query GT visibility is respected by
            # falling back to the frame-0 anchor when a query is occluded at
            # the sampled frame.
            gen = torch.Generator(device=device)
            gen.manual_seed(int(epoch * 10007 + 13))
            for _k in range(K_bank - 1):
                t_anchor = int(torch.randint(
                    1, max(T, 2), (1,), generator=gen, device=device,
                ).item())
                xy_k = tracks_gt[:, :, t_anchor, :]  # [B, Q, 2]
                # Gate to GT-visible queries at this anchor frame.
                vis_k = (vis_gt[:, :, t_anchor] > 0.5)  # [B, Q]
                desc_k = correspondence.sample_embeddings_at_points(
                    descriptor_maps[t_anchor], xy_k, patch_size,
                )  # [B, Q, C_desc]
                fine_k = correspondence.sample_embeddings_at_points(
                    fine_feature_maps[:, t_anchor], xy_k, feature_stride,
                )  # [B, Q, C_f]
                # Replace invisible queries with the frame-0 anchor.
                vis_k_e = vis_k.unsqueeze(-1).to(dtype=desc_k.dtype)
                desc_k = vis_k_e * desc_k + (1.0 - vis_k_e) * query_desc_0
                fine_k = vis_k_e * fine_k + (1.0 - vis_k_e) * query_fine_0
                desc_bank_list.append(desc_k)
                fine_bank_list.append(fine_k)
            query_desc_bank = torch.stack(desc_bank_list, dim=2)  # [B, Q, K, C_desc]
            query_fine_bank = torch.stack(fine_bank_list, dim=2)  # [B, Q, K, C_f]
        else:
            query_desc_bank = query_desc_0   # [B, Q, C_desc]
            query_fine_bank = query_fine_0   # [B, Q, C_f]

        # --- Forward tracking ---
        if use_long and T > T_win:
            # Chained sub-window refinement. The backbone has already produced
            # all ``T`` maps; we slice them and run the refinement on each
            # sub-window of size ``T_win``. Gradients flow through refinement
            # only (backbone is frozen) so memory scales with (sub-window x Q)
            # and we get *direct* supervision for accumulated drift and for
            # reappearance events that straddle sub-window boundaries.
            sub_size = int(T_win)
            # Non-overlapping sub-windows for a clean per-frame loss reduction.
            sub_starts = list(range(0, T, sub_size))
            sub_tracks_list = []
            sub_vis_list = []
            cur_desc_bank = query_desc_bank
            cur_fine_bank = query_fine_bank
            for s_start in sub_starts:
                s_end = min(s_start + sub_size, T)
                if s_end - s_start < 2:
                    break
                sub_desc_maps = descriptor_maps[s_start:s_end]
                sub_fine = fine_feature_maps[:, s_start:s_end]
                sub_coarse = self._global_coarse_match(
                    query_points, sub_desc_maps, query_descriptors=cur_desc_bank,
                )  # [B, Q, t_sub, 2]
                sub_out = self.refinement_net(
                    sub_coarse, cur_fine_bank, sub_fine, feature_stride,
                )
                sub_tracks_list.append(sub_out["tracks"])
                sub_vis_list.append(sub_out["visibility"])

                # Refresh the bank from a GT-visible frame inside this
                # sub-window so the *next* sub-window starts with appearance
                # templates that reflect the current photometric conditions.
                if (
                    K_bank > 1
                    and use_pseudo
                    and tracks_gt is not None
                    and vis_gt is not None
                    and s_end < T
                ):
                    local_vis = vis_gt[:, :, s_start:s_end]  # [B, Q, t_sub]
                    # Prefer the latest visible frame in the window per query.
                    _, best_local = local_vis.flip(dims=[-1]).max(dim=-1)
                    best_local = (local_vis.shape[-1] - 1 - best_local)  # latest max
                    best_global = s_start + best_local  # [B, Q]
                    refresh = local_vis.gather(-1, best_local.unsqueeze(-1)).squeeze(-1) > 0.5
                    if refresh.any():
                        # Gather tracks_gt at (b, q, best_global[b,q]).
                        idx_expand = best_global.view(B, -1, 1, 1).expand(B, -1, 1, 2)
                        xy_k = tracks_gt.gather(2, idx_expand).squeeze(2)  # [B, Q, 2]
                        # Per unique frame index, sample descriptor + fine.
                        unique_t, inv = torch.unique(best_global, return_inverse=True)
                        new_desc_k = query_desc_0.clone()
                        new_fine_k = query_fine_0.clone()
                        for ut_i, ut in enumerate(unique_t.tolist()):
                            sel = (inv == ut_i) & refresh  # [B, Q]
                            if not sel.any():
                                continue
                            # Build per-batch-item selected positions via a
                            # masked scatter; sample_embeddings_at_points needs
                            # [B, N, 2] so we run it with xy masked and then
                            # copy selected entries into new_desc_k / fine_k.
                            desc_map_t = descriptor_maps[ut]  # [B, C, H_p, W_p]
                            fine_map_t = fine_feature_maps[:, ut]  # [B, C, H_f, W_f]
                            # Sample all queries at their (b,q) position
                            d_all = correspondence.sample_embeddings_at_points(
                                desc_map_t, xy_k, patch_size,
                            )  # [B, Q, C_desc]
                            f_all = correspondence.sample_embeddings_at_points(
                                fine_map_t, xy_k, feature_stride,
                            )  # [B, Q, C_f]
                            # Only copy those entries where ``sel`` is True.
                            sel_e = sel.unsqueeze(-1)
                            new_desc_k = torch.where(sel_e, d_all, new_desc_k)
                            new_fine_k = torch.where(sel_e, f_all, new_fine_k)
                        # Round-robin: rotate bank slots, write new anchor into
                        # the oldest non-frame-0 slot (slot 1 modulo K-1 + 1).
                        # We don't need a pointer here since sub-windows are
                        # few; simple scheme: push into slot ((s_idx % (K-1)) + 1).
                        slot = ((sub_starts.index(s_start)) % (K_bank - 1)) + 1
                        cur_desc_bank = cur_desc_bank.clone()
                        cur_fine_bank = cur_fine_bank.clone()
                        cur_desc_bank[:, :, slot, :] = new_desc_k
                        cur_fine_bank[:, :, slot, :] = new_fine_k

            tracks_fwd = torch.cat(sub_tracks_list, dim=2)   # [B, Q, T, 2]
            vis_fwd = torch.cat(sub_vis_list, dim=2)         # [B, Q, T]
            # Keep a coarse-tracks proxy for debug returns; reuse the first sub.
            coarse_tracks = sub_tracks_list[0].detach()
        else:
            coarse_tracks = self._global_coarse_match(
                query_points, descriptor_maps, query_descriptors=query_desc_bank,
            )

            fwd_out = self.refinement_net(
                coarse_tracks, query_fine_bank, fine_feature_maps, feature_stride,
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
            # Second full refinement forward stacks autograd state on top of the
            # first; checkpoint trades extra compute for much lower peak VRAM.
            ref_mod = unwrap_model(self.refinement_net)
            fs = int(feature_stride)

            def _cycle_refine_tracks(
                co: torch.Tensor,
                qf: torch.Tensor,
                fm: torch.Tensor,
            ) -> torch.Tensor:
                return ref_mod(co, qf, fm, fs)["tracks"]  # [B, Q, T, 2]

            tracks_bwd_tensor = checkpoint(
                _cycle_refine_tracks,
                coarse_rev,
                rev_query_fine,
                fine_rev,
                use_reentrant=False,
            )
            tracks_bwd = tracks_bwd_tensor.flip(dims=[2])  # [B, Q, T, 2]
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

        loss_self_sup_tensor = loss_dict["loss_total"]
        metrics_self_sup: dict = dict(loss_dict.pop("metrics"))
        metrics_self_sup["loss_self_sup_total"] = float(loss_self_sup_tensor.detach().item())

        metrics_pseudo_gt: dict = {
            "pseudo_gt_active": 0.0,
            "pseudo_lambda": 0.0,
            "loss_sup_pos": 0.0,
            "loss_sup_vis": 0.0,
            "loss_sup_total": 0.0,
            "sup_mask_fraction": 0.0,
        }
        loss_total = loss_self_sup_tensor

        if use_pseudo and tracks_gt is not None and vis_target is not None and composite_m is not None:
            sup = temporal_supervised_losses(
                tracks_pred=tracks_fwd,
                visibility_pred=vis_fwd,
                tracks_gt=tracks_gt,
                composite_mask=composite_m,
                vis_target=vis_target,
                config=config,
                position_mask=pos_mask_m,
                vis_gt=vis_gt,
            )
            lam_sched = config.get("_SCHED_PSEUDO_SUP_LAMBDA", None)
            if lam_sched is not None:
                lam = float(lam_sched)
            else:
                cur_epochs = max(1, int(config.get("PSEUDO_GT_CURRICULUM_EPOCHS", 10)))
                lam = lam_max * min(1.0, float(epoch + 1) / float(cur_epochs))
            loss_total = (1.0 - lam) * loss_self_sup_tensor + lam * sup["loss_sup_total"]
            metrics_pseudo_gt.update(sup["metrics"])
            metrics_pseudo_gt["pseudo_lambda"] = float(lam)
            metrics_pseudo_gt["pseudo_gt_active"] = 1.0

        loss_dict["loss_total"] = loss_total
        loss_dict["metrics_self_sup"] = metrics_self_sup
        loss_dict["metrics_pseudo_gt"] = metrics_pseudo_gt

        return loss_dict
