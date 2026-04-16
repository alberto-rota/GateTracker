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
    ):
        """
        Args:
            feature_dim:  Channel dimension of fine feature maps from the backbone.
            hidden_dim:   Hidden dimension of the temporal mixer.
            num_blocks:   Number of TemporalMixerBlocks per iteration.
            num_iters:    Number of refinement iterations (weight-shared).
            corr_radius:  Radius of the local correlation window (side = 2r+1).
            kernel_size:  Temporal convolution kernel size.
            pos_encoding_bands: Fourier encoding frequency bands.
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_iters = num_iters
        self.corr_radius = corr_radius
        self.pos_encoding_bands = pos_encoding_bands

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

    def _extract_local_correlation(
        self,
        query_features: torch.Tensor,
        feature_maps: torch.Tensor,
        positions: torch.Tensor,
        feature_stride: int,
    ) -> torch.Tensor:
        """Extract local correlation volumes around predicted positions.

        Args:
            query_features: [B, Q, C] query descriptors from frame 0.
            feature_maps:   [B, T, C, H_f, W_f] fine feature maps.
            positions:      [B, Q, T, 2] current position estimates (pixel coords).
            feature_stride: Pixel stride of the feature map.

        Returns:
            [B, Q, T, W_w^2] local correlation volumes.
        """
        B, T, C, H_f, W_f = feature_maps.shape
        Q = query_features.shape[1]
        r = self.corr_radius
        W_w = 2 * r + 1

        # Convert pixel positions to feature coordinates
        feat_coords = positions / feature_stride  # [B, Q, T, 2]

        # Build local offset grid: [W_w^2, 2]
        offsets = torch.arange(-r, r + 1, device=positions.device, dtype=positions.dtype)
        oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
        offset_grid = torch.stack([ox.reshape(-1), oy.reshape(-1)], dim=-1)  # [W_w^2, 2]

        # Sample positions: [B, Q, T, W_w^2, 2]
        sample_coords = feat_coords.unsqueeze(-2) + offset_grid  # [B, Q, T, W_w^2, 2]

        # Normalize to [-1, 1] for grid_sample
        sample_norm = sample_coords.clone()
        sample_norm[..., 0] = sample_norm[..., 0] / max(W_f - 1, 1) * 2 - 1
        sample_norm[..., 1] = sample_norm[..., 1] / max(H_f - 1, 1) * 2 - 1

        # Reshape for grid_sample: [B*T, C, H_f, W_f] with grid [B*T, Q*W_w^2, 1, 2]
        fmaps_flat = feature_maps.reshape(B * T, C, H_f, W_f)
        grid_flat = sample_norm.permute(0, 2, 1, 3, 4).reshape(B * T, Q * W_w * W_w, 1, 2)

        sampled = F.grid_sample(
            fmaps_flat, grid_flat, mode="bilinear", align_corners=True,
        )  # [B*T, C, Q*W_w^2, 1]
        sampled = sampled.squeeze(-1).reshape(B, T, C, Q, W_w * W_w)  # [B, T, C, Q, W_w^2]
        sampled = sampled.permute(0, 3, 1, 4, 2)  # [B, Q, T, W_w^2, C]

        # Dot product with query features
        query_exp = query_features.unsqueeze(2).unsqueeze(3)  # [B, Q, 1, 1, C]
        corr = (sampled * query_exp).sum(dim=-1)  # [B, Q, T, W_w^2]

        return corr

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
            # 1. Local correlation
            corr = self._extract_local_correlation(
                query_features, feature_maps, current_tracks, feature_stride,
            )  # [B, Q, T, W_w^2]

            # 2. Position encoding
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

            # Reshape to [BQ, T, D] for mixer blocks
            x = mixer_input.reshape(B * Q, T, -1)
            x = self.input_proj(x)  # [BQ, T, hidden_dim]

            for block in self.blocks:
                x = block(x)  # [BQ, T, hidden_dim]

            hidden = x  # [BQ, T, hidden_dim]

            # 4. Predict delta and visibility
            delta = self.delta_head(x).reshape(B, Q, T, 2)  # [B, Q, T, 2]
            vis_logit = self.vis_head(x).reshape(B, Q, T)  # [B, Q, T]

            current_tracks = current_tracks + delta
            all_vis_logits.append(vis_logit)

        # Average visibility logits across iterations
        visibility = torch.stack(all_vis_logits, dim=0).mean(dim=0)  # [B, Q, T]

        return {
            "tracks": current_tracks,       # [B, Q, T, 2]
            "visibility": visibility,        # [B, Q, T]
        }
