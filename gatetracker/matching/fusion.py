import torch
import torch.nn as nn
import torch.nn.functional as F

from gatetracker.matching import correspondence
from gatetracker.utils.tensor_ops import embedding2chw


def _group_norm_groups(num_channels: int) -> int:
    for num_groups in (8, 4, 2, 1):
        if num_channels % num_groups == 0:
            return num_groups
    return 1


class RegisterGatedHierarchicalFusion(nn.Module):
    """
    Dynamically fuses DINOv3 layers using both local patch evidence and
    global register-token priors.
    """

    def __init__(
        self,
        hidden_dim,
        layer_indices,
        num_register_tokens=4,
        gate_temperature=2.0,
        uniform_mixing=0.1,
        layer_dropout_p=0.1,
        gate_logit_bound=1.5,
    ):
        super().__init__()
        self.layer_indices = list(layer_indices)
        self.num_register_tokens = num_register_tokens
        self.gate_temperature = gate_temperature
        self.uniform_mixing = uniform_mixing
        self.layer_dropout_p = layer_dropout_p
        self.gate_logit_bound = gate_logit_bound
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim, bias=False),
                )
                for _ in self.layer_indices
            ]
        )
        self.local_gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, 1, bias=False),
                )
                for _ in self.layer_indices
            ]
        )
        self.register_gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, 1, bias=False),
                )
                for _ in self.layer_indices
            ]
        )
        for gate_module in list(self.local_gates) + list(self.register_gates):
            nn.init.zeros_(gate_module[-1].weight)
        self.output_refine = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.register_buffer(
            "_layer_index_buf",
            torch.tensor(self.layer_indices, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, hidden_states, return_diagnostics=False):
        projected_layers = []
        gate_logits = []

        for module_idx, layer_idx in enumerate(self.layer_indices):
            layer_hidden = hidden_states[layer_idx]  # [B, N_tokens, C]
            patch_tokens = layer_hidden[:, 1 + self.num_register_tokens :, :]  # [B, N, C]
            register_tokens = layer_hidden[:, 1 : 1 + self.num_register_tokens, :]  # [B, R, C]

            projected = self.projections[module_idx](patch_tokens)  # [B, N, C]
            local_gate = self.local_gates[module_idx](projected)  # [B, N, 1]

            if register_tokens.shape[1] > 0:
                register_context = register_tokens.mean(dim=1)  # [B, C]
            else:
                register_context = layer_hidden[:, 0, :]  # [B, C]
            register_gate = self.register_gates[module_idx](register_context).unsqueeze(1)  # [B, 1, 1]

            projected_layers.append(projected)
            gate_logits.append(
                torch.tanh(local_gate + register_gate) * self.gate_logit_bound
            )

        stacked_projected = torch.stack(projected_layers, dim=0)  # [L, B, N, C]
        stacked_logits = torch.stack(gate_logits, dim=0)  # [L, B, N, 1]
        stacked_logits = stacked_logits - stacked_logits.mean(
            dim=0, keepdim=True
        )  # [L, B, N, 1]
        if self.training and self.layer_dropout_p > 0.0 and len(self.layer_indices) > 1:
            keep_mask = (
                torch.rand(
                    len(self.layer_indices), 1, 1, 1, device=stacked_logits.device
                )
                > self.layer_dropout_p
            )
            if not keep_mask.any():
                keep_mask[torch.randint(len(self.layer_indices), (1,))] = True
            stacked_logits = stacked_logits.masked_fill(~keep_mask, -1e4)
        layer_weights = torch.softmax(
            stacked_logits / max(self.gate_temperature, 1e-6), dim=0
        )  # [L, B, N, 1]
        if self.uniform_mixing > 0.0:
            layer_weights = (
                (1.0 - self.uniform_mixing) * layer_weights
                + self.uniform_mixing / len(self.layer_indices)
            )
        fused = (layer_weights * stacked_projected).sum(dim=0)  # [B, N, C]
        fused = fused + self.output_refine(fused)

        if not return_diagnostics:
            return fused

        layer_weight_maps = layer_weights.squeeze(-1).permute(1, 0, 2).contiguous()  # [B, L, N]
        effective_layer = (
            layer_weight_maps * self._layer_index_buf.view(1, -1, 1)
        ).sum(dim=1)  # [B, N]
        diagnostics = {
            "layer_weights": layer_weight_maps,
            "effective_layer": effective_layer,
            "max_weight": layer_weight_maps.max(dim=1).values,
            "layer_indices": self._layer_index_buf,
            "stacked_projected": stacked_projected,  # [L, B, N, C]
            "layer_weights_LBN1": layer_weights,  # [L, B, N, 1]
        }
        return fused, diagnostics


def build_gated_layer_maps(diagnostics: dict) -> dict:
    """
    Reshape per-layer intermediates from RegisterGatedHierarchicalFusion
    diagnostics into spatial feature maps suitable for ``grid_sample``.

    Args:
        diagnostics: dict from ``hierarchical_fusion(..., return_diagnostics=True)``
                     that must contain ``stacked_projected`` and ``layer_weights_LBN1``.

    Returns:
        dict with:
            ``projected_maps``:   [L, B, C, H_p, W_p]
            ``gate_weight_maps``: [L, B, 1, H_p, W_p]
        or ``None`` if the required keys are absent.
    """
    if diagnostics is None:
        return None
    stacked_projected = diagnostics.get("stacked_projected")  # [L, B, N, C]
    layer_weights = diagnostics.get("layer_weights_LBN1")  # [L, B, N, 1]
    if stacked_projected is None or layer_weights is None:
        return None

    L, B, N, C = stacked_projected.shape
    projected_maps = torch.stack(
        [embedding2chw(stacked_projected[l]) for l in range(L)], dim=0,
    )  # [L, B, C, H_p, W_p]

    H_p, W_p = projected_maps.shape[-2], projected_maps.shape[-1]
    gate_weight_maps = layer_weights.permute(0, 1, 3, 2).reshape(
        L, B, 1, H_p, W_p,
    )  # [L, B, 1, H_p, W_p]

    return {
        "projected_maps": projected_maps,
        "gate_weight_maps": gate_weight_maps,
    }


def gated_or_standard_query(
    model_output: dict,
    query_points: torch.Tensor,
    patch_size: int,
    gated_layer_maps: dict,
    output_refine: nn.Module,
) -> torch.Tensor:
    """
    Sample query embeddings at arbitrary pixel locations, using gated
    per-layer interpolation when available, falling back to standard
    bilinear sampling from the fused map.

    Args:
        model_output:     dict with ``"source_embedding_match"`` key ([B, C, N]).
        query_points:     [B, Q, 2] pixel coordinates (x, y).
        patch_size:       patch stride in pixels.
        gated_layer_maps: output of ``build_gated_layer_maps`` or None.
        output_refine:    ``hierarchical_fusion.output_refine`` module.

    Returns:
        [B, Q, C] L2-normalized query embeddings.
    """
    if gated_layer_maps is not None:
        return correspondence.sample_gated_embeddings_at_points(
            gated_layer_maps["projected_maps"],
            gated_layer_maps["gate_weight_maps"],
            query_points,
            patch_size,
            output_refine,
        )
    source_map = embedding2chw(
        model_output["source_embedding_match"], embed_dim_last=False,
    )  # [B, C, H_p, W_p]
    return correspondence.sample_embeddings_at_points(
        source_map, query_points, patch_size,
    )
