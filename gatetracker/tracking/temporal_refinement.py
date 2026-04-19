"""
TAPIR-style iterative temporal refinement network for point tracking.

Operates **per-point independently** across all T frames simultaneously.
Takes coarse track estimates from global descriptor matching and refines
them through K iterations of local correlation + temporal mixing.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _fourier_position_encoding(
    coords: torch.Tensor,
    num_bands: int = 16,
    max_freq: float = 512.0,
) -> torch.Tensor:
    """Sinusoidal Fourier encoding of continuous coordinates.

    Args:
        coords: [..., D] arbitrary coordinate tensor.
        num_bands: Number of frequency bands per dimension.
        max_freq: Maximum frequency.

    Returns:
        [..., D * num_bands * 2 + D] encoded positions (sin + cos + raw).
    """
    freqs = torch.linspace(
        0.0, math.log2(max_freq), num_bands, device=coords.device, dtype=coords.dtype,
    )
    freqs = (2.0 ** freqs) * math.pi  # [num_bands]

    # coords: [..., D], freqs: [num_bands] -> [..., D, num_bands]
    x = coords.unsqueeze(-1) * freqs  # [..., D, num_bands]
    enc = torch.cat([x.sin(), x.cos()], dim=-1)  # [..., D, num_bands*2]
    enc = enc.flatten(-2)  # [..., D * num_bands * 2]
    return torch.cat([coords, enc], dim=-1)  # [..., D * (num_bands*2 + 1)]


class TemporalMixerBlock(nn.Module):
    """Single mixer block: depthwise temporal conv + channel MLP, with residuals."""

    def __init__(self, hidden_dim: int, kernel_size: int = 5):
        super().__init__()
        padding = kernel_size // 2
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.temporal_conv = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=kernel_size,
            padding=padding, groups=hidden_dim,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [BQ, T, D]
        Returns:
            [BQ, T, D]
        """
        # Temporal conv (operates over T dimension)
        h = self.norm1(x)
        h = h.permute(0, 2, 1)  # [BQ, D, T]
        h = self.temporal_conv(h)  # [BQ, D, T]
        h = h.permute(0, 2, 1)  # [BQ, T, D]
        x = x + h

        # Channel MLP
        x = x + self.channel_mlp(self.norm2(x))
        return x


class TemporalRefinementNetwork(nn.Module):
    """TAPIR/PIPs-style iterative temporal refinement for point tracking.

    Given coarse track positions from global descriptor matching, this module
    iteratively refines positions by extracting local correlation features at
    each frame and mixing them through temporal convolutions.

    All operations are per-point independent (no cross-point interactions),
    so the network scales to arbitrary numbers of query points.
    """

    def __init__(
        self,
        feature_dim: int = 64,
        hidden_dim: int = 128,
        num_blocks: int = 4,
        num_iters: int = 4,
        corr_radius: int = 3,
        kernel_size: int = 5,
        pos_encoding_bands: int = 16,
        softmax_temperature: float = 0.1,
    ):
        """
        Args:
            feature_dim:         Channel dimension of fine feature maps from the backbone.
            hidden_dim:          Hidden dimension of the temporal mixer.
            num_blocks:          Number of TemporalMixerBlocks per iteration.
            num_iters:           Number of refinement iterations (weight-shared).
            corr_radius:         Radius of the local correlation window (side = 2r+1).
            kernel_size:         Temporal convolution kernel size.
            pos_encoding_bands:  Fourier encoding frequency bands.
            softmax_temperature: Temperature for the deterministic soft-argmax
                sub-pixel offset applied at every iteration. Mirrors the matching
                refiner's ``FINE_REFINEMENT_TEMPERATURE`` so the temporal network
                inherits the same continuous (non-snap) localization.
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_iters = num_iters
        self.corr_radius = corr_radius
        self.pos_encoding_bands = pos_encoding_bands
        self.softmax_temperature = float(softmax_temperature)

        corr_window = (2 * corr_radius + 1) ** 2
        # xy (2 dims) + time (1 dim) → each with (2*bands + 1) encoding
        pos_dim = 3 * (2 * pos_encoding_bands + 1)
        input_dim = corr_window + pos_dim + hidden_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            TemporalMixerBlock(hidden_dim, kernel_size) for _ in range(num_blocks)
        ])
        self.delta_head = nn.Linear(hidden_dim, 2)
        self.vis_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    # Soft cap on the number of elements in the per-call ``grid_sample``
    # output. Above ~INT_MAX = 2**31 the cuDNN ``grid_sampler`` kernel raises
    # ``CUDNN_STATUS_NOT_SUPPORTED``; beyond that the intermediate also
    # dominates VRAM (e.g. B=32, T=24, Q=1024, W_w=9, C=64 needs ~16 GB just
    # for ``sampled``). We chunk along the query axis to stay well below
    # both limits while preserving exact equivalence (per-point op is
    # independent across queries).
    #
    # During autograd, many chunks' activations coexist in the graph, and a
    # second refinement forward (cycle loss) stacks further peaks — keep this
    # budget conservative so each ``sampled`` slab stays smaller (helps both
    # allocator fragmentation and peak VRAM).
    _GRID_SAMPLE_MAX_ELEMENTS: int = 1 << 24  # 16_777_216 elements per chunk

    def _extract_local_correlation(
        self,
        query_features: torch.Tensor,
        feature_maps: torch.Tensor,
        positions: torch.Tensor,
        feature_stride: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract local correlation volumes and a soft-argmax sub-pixel offset.

        The soft-argmax offset mirrors ``feature_softargmax_refiner`` used by the
        matching pipeline: it is a deterministic, fully differentiable continuous
        sub-pixel shift in pixel units, computed from the local correlation over
        the fine feature map. Feeding it back into ``current_tracks`` breaks the
        16-px patch-grid "snap" that arises when the learned delta head still
        outputs ~0 (e.g. early in training).

        For long-horizon dense tracking (large ``Q``, large ``W_w``, long ``T``)
        the intermediate sampled tensor would exceed cuDNN's INT_MAX limit, so
        the work is automatically chunked along the query axis.

        Args:
            query_features: [B, Q, C] query descriptors from frame 0
                (already L2-normalized by ``sample_embeddings_at_points``).
            feature_maps:   [B, T, C, H_f, W_f] fine feature maps.
            positions:      [B, Q, T, 2] current position estimates (pixel coords).
            feature_stride: Pixel stride of the feature map.

        Returns:
            corr:              [B, Q, T, W_w^2] raw correlation features for the mixer.
            soft_argmax_delta: [B, Q, T, 2] expected sub-pixel shift in pixel units.
        """
        B, T, C, H_f, W_f = feature_maps.shape
        Q = query_features.shape[1]
        r = self.corr_radius
        W_w = 2 * r + 1

        offsets = torch.arange(-r, r + 1, device=positions.device, dtype=positions.dtype)
        oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
        offset_grid = torch.stack([ox.reshape(-1), oy.reshape(-1)], dim=-1)  # [W_w^2, 2]

        # Choose Q chunk size so each ``grid_sample`` output stays under the
        # element budget. The per-chunk output is [B*T, C, q*W_w^2, 1].
        elems_per_query = max(1, B * T * C * W_w * W_w)
        q_chunk = max(1, min(Q, self._GRID_SAMPLE_MAX_ELEMENTS // elems_per_query))

        fmaps_flat = feature_maps.reshape(B * T, C, H_f, W_f).contiguous()
        offset_pixels = offset_grid * float(feature_stride)  # [W_w^2, 2]
        offset_pixels_view = offset_pixels.view(1, 1, 1, -1, 2)
        inv_temp = 1.0 / max(self.softmax_temperature, 1e-6)

        corr_chunks = []
        delta_chunks = []
        for q_start in range(0, Q, q_chunk):
            q_end = min(q_start + q_chunk, Q)
            qf = query_features[:, q_start:q_end]              # [B, q, C]
            pos = positions[:, q_start:q_end]                   # [B, q, T, 2]

            feat_coords = pos / feature_stride                  # [B, q, T, 2]
            sample_coords = feat_coords.unsqueeze(-2) + offset_grid  # [B, q, T, W_w^2, 2]

            sample_norm = torch.empty_like(sample_coords)
            sample_norm[..., 0] = sample_coords[..., 0] / max(W_f - 1, 1) * 2 - 1
            sample_norm[..., 1] = sample_coords[..., 1] / max(H_f - 1, 1) * 2 - 1

            q = q_end - q_start
            grid_flat = (
                sample_norm.permute(0, 2, 1, 3, 4)
                .contiguous()
                .reshape(B * T, q * W_w * W_w, 1, 2)
            )

            sampled = F.grid_sample(
                fmaps_flat, grid_flat, mode="bilinear", align_corners=True,
            )  # [B*T, C, q*W_w^2, 1]
            sampled = sampled.squeeze(-1).reshape(B, T, C, q, W_w * W_w)
            sampled = sampled.permute(0, 3, 1, 4, 2)  # [B, q, T, W_w^2, C]

            query_exp = qf.unsqueeze(2).unsqueeze(3)  # [B, q, 1, 1, C]
            corr_chunk = (sampled * query_exp).sum(dim=-1)  # [B, q, T, W_w^2]

            q_norm = F.normalize(qf, dim=-1).unsqueeze(2).unsqueeze(3)
            s_norm = F.normalize(sampled, dim=-1)
            logits = (s_norm * q_norm).sum(dim=-1)  # [B, q, T, W_w^2]
            probs = torch.softmax(logits * inv_temp, dim=-1)
            soft_argmax_delta_chunk = (
                probs.unsqueeze(-1) * offset_pixels_view
            ).sum(dim=-2)  # [B, q, T, 2]

            corr_chunks.append(corr_chunk)
            delta_chunks.append(soft_argmax_delta_chunk)
            del sampled, sample_norm, sample_coords, grid_flat, logits, probs

        if len(corr_chunks) == 1:
            return corr_chunks[0], delta_chunks[0]
        return torch.cat(corr_chunks, dim=1), torch.cat(delta_chunks, dim=1)

    def forward(
        self,
        coarse_tracks: torch.Tensor,
        query_features: torch.Tensor,
        feature_maps: torch.Tensor,
        feature_stride: int,
    ) -> dict:
        """Run iterative temporal refinement.

        Args:
            coarse_tracks:  [B, Q, T, 2] initial position estimates from global matching.
            query_features: [B, Q, C] fine feature descriptors at query positions (frame 0).
            feature_maps:   [B, T, C, H_f, W_f] fine feature maps for all frames.
            feature_stride: Pixel stride of the fine feature map.

        Returns:
            dict with:
                tracks:     [B, Q, T, 2] refined pixel coordinates.
                visibility: [B, Q, T] visibility logits.
        """
        B, Q, T, _ = coarse_tracks.shape
        device = coarse_tracks.device
        dtype = coarse_tracks.dtype

        current_tracks = coarse_tracks.clone()  # [B, Q, T, 2]

        # Temporal index encoding: [1, 1, T, 1]
        t_idx = torch.linspace(0, 1, T, device=device, dtype=dtype).view(1, 1, T, 1)

        # Initialize hidden state
        hidden = torch.zeros(B * Q, T, self.hidden_dim, device=device, dtype=dtype)

        all_vis_logits = []
        for _iter in range(self.num_iters):
            # 1. Local correlation + deterministic soft-argmax sub-pixel offset.
            #    ``soft_argmax_delta`` is the exact same quantity that the matching
            #    refiner (``feature_softargmax_refiner``) uses, just evaluated at
            #    every frame. Applying it unconditionally gives continuous (non-
            #    snap) localization even when ``delta_head`` still outputs ~0.
            corr, soft_argmax_delta = self._extract_local_correlation(
                query_features, feature_maps, current_tracks, feature_stride,
            )  # corr: [B, Q, T, W_w^2], soft_argmax_delta: [B, Q, T, 2]
            current_tracks = current_tracks + soft_argmax_delta

            # 2. Position encoding (use the post-softargmax position)
            pos_input = torch.cat([
                current_tracks,  # [B, Q, T, 2]
                t_idx.expand(B, Q, -1, -1),  # [B, Q, T, 1]
            ], dim=-1)  # [B, Q, T, 3]
            pos_enc = _fourier_position_encoding(
                pos_input, num_bands=self.pos_encoding_bands,
            )  # [B, Q, T, pos_dim]

            # 3. Concatenate inputs
            mixer_input = torch.cat([
                corr,
                pos_enc,
                hidden.reshape(B, Q, T, self.hidden_dim),
            ], dim=-1)  # [B, Q, T, input_dim]

            x = mixer_input.reshape(B * Q, T, -1)  # [BQ, T, input_dim]
            x = self.input_proj(x)                 # [BQ, T, hidden_dim]

            for block in self.blocks:
                x = block(x)  # [BQ, T, hidden_dim]

            hidden = x  # [BQ, T, hidden_dim]

            # 4. Residual delta on top of the soft-argmax position and visibility.
            delta = self.delta_head(x).reshape(B, Q, T, 2)  # [B, Q, T, 2]
            vis_logit = self.vis_head(x).reshape(B, Q, T)   # [B, Q, T]

            current_tracks = current_tracks + delta
            all_vis_logits.append(vis_logit)

        # Average visibility logits across iterations
        visibility = torch.stack(all_vis_logits, dim=0).mean(dim=0)  # [B, Q, T]

        return {
            "tracks": current_tracks,       # [B, Q, T, 2]
            "visibility": visibility,        # [B, Q, T]
        }
