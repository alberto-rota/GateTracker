"""
Gated bilinear tracking integration.

Provides helper functions and a mixin for wiring the fuse-after-interpolate
gated tracking into MatcherModel and Matcher.

INTEGRATION INSTRUCTIONS
========================
Apply the following changes to ``matching/matching.py``:

1) In ``RegisterGatedHierarchicalFusion.forward()``, add two keys to the
   ``diagnostics`` dict (right after the existing "layer_indices" key):

       diagnostics = {
           "layer_weights": layer_weight_maps,
           "effective_layer": effective_layer,
           "max_weight": layer_weight_maps.max(dim=1).values,
           "layer_indices": layer_index_tensor,
+          "stacked_projected": stacked_projected,   # [L, B, N, C]
+          "layer_weights_LBN1": layer_weights,       # [L, B, N, 1]
       }

2) In ``MatcherModel._encode_image()``, after the hierarchical_fusion call,
   add a call to ``build_gated_layer_maps``:

       matched_patch_tokens, diagnostics = self.hierarchical_fusion(
           features, return_diagnostics=True
       )
+      gated_layer_maps = build_gated_layer_maps(diagnostics)

   Then return ``gated_layer_maps`` as a 6th element of the return tuple:

       return (
           raw_patch_tokens,
           matched_patch_tokens,
           cls_token,
           diagnostics,
           fine_feature_map,
+          gated_layer_maps,   # dict or None
       )

   For the DINOv2 path, return None as the 6th element.

3) In ``MatcherModel.forward()``, capture the 6th return value and store it:

       (
           source_raw, source_matched, source_cls,
           source_diag, source_fine,
+          source_gated_maps,
       ) = self._encode_image(source)
       ...same for target...

+      self.latest_gated_layer_maps = {
+          "source": source_gated_maps,
+          "target": target_gated_maps,
+      }

4) In ``track_points()``, after the model forward pass, replace the
   coarse query embedding sampling block with:

       gated_maps = getattr(self.model, "latest_gated_layer_maps", {})
       src_gated = gated_maps.get("source") if gated_maps else None

       if src_gated is not None:
           query_embs = correspondence.sample_gated_embeddings_at_points(
               src_gated["projected_maps"],
               src_gated["gate_weight_maps"],
               query_points,
               self.patch_size,
               self.model.hierarchical_fusion.output_refine,
           )
       else:
           source_map = embedding2chw(
               modeloutput["source_embedding_match"], embed_dim_last=False
           )
           query_embs = correspondence.sample_embeddings_at_points(
               source_map, query_points, self.patch_size
           )

5) In ``track_points_window()``, cache gated layer maps alongside embeddings:

       gated_maps_list = []
       with torch.no_grad():
           for t in range(T):
               (
                   _, matched_tokens, _, _, fine_map,
+                  gated_maps_t,
               ) = self.model._encode_image(frames[:, t])
               ...
+              gated_maps_list.append(gated_maps_t)

   Then in the per-frame loop, use ``gated_maps_list[t-1]`` for gated sampling.
"""

import torch
import torch.nn as nn

from matching import correspondence
from utilities.tensor_utils import embedding2chw


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
