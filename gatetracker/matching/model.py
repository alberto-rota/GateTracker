import os
import pickle
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F

from gatetracker.backbone.feature_extractor import FeatureExtractor
from gatetracker.matching.fusion import (
    RegisterGatedHierarchicalFusion,
    build_gated_layer_maps,
    _group_norm_groups,
)

from gatetracker.utils.logger import get_logger
from gatetracker.utils.tensor_ops import chw2embedding, embedding2chw
from utilities.dev_utils import download_from_gcs

logger = get_logger(__name__).set_context("MATCHING")


def _load_checkpoint_mapping(path: str, map_location):
    """Load checkpoint dict; prefer safe unpickling, fall back for legacy NumPy metadata."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError as exc:
        msg = str(exc)
        if "Weights only load failed" in msg or "Unsupported global" in msg:
            return torch.load(path, map_location=map_location, weights_only=False)
        raise


class LocalRefinementFeatureHead(nn.Module):
    """
    Builds localizable fine features from RGB appearance and DINO context.
    """

    def __init__(
        self,
        image_channels: int,
        descriptor_dim: int,
        output_dim: int,
        feature_stride: int = 4,
    ):
        super().__init__()
        hidden_dim = max(output_dim // 2, 16)
        self.feature_stride = max(1, int(feature_stride))
        self.rgb_stem = nn.Sequential(
            nn.Conv2d(image_channels, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_norm_groups(hidden_dim), hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_norm_groups(output_dim), output_dim),
            nn.GELU(),
        )
        self.coarse_projection = nn.Sequential(
            nn.Conv2d(descriptor_dim, output_dim, kernel_size=1, bias=False),
            nn.GroupNorm(_group_norm_groups(output_dim), output_dim),
            nn.GELU(),
        )
        self.context_projection = nn.Sequential(
            nn.Conv2d(descriptor_dim, output_dim, kernel_size=1, bias=False),
            nn.GroupNorm(_group_norm_groups(output_dim), output_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(output_dim * 3, output_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_norm_groups(output_dim), output_dim),
            nn.GELU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1, bias=False),
        )

    def forward(
        self,
        image: torch.Tensor,
        coarse_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            image: [B, 3, H, W] RGB image.
            coarse_tokens: [B, C, N] or [B, N, C] descriptor tokens.
            context_tokens: [B, C, N] or [B, N, C] lower-level context tokens.

        Returns:
            [B, C_f, H / s, W / s] normalized fine features.
        """
        target_height = max(image.shape[-2] // self.feature_stride, 1)
        target_width = max(image.shape[-1] // self.feature_stride, 1)
        coarse_map = embedding2chw(coarse_tokens, embed_dim_last=False)
        context_map = embedding2chw(context_tokens, embed_dim_last=False)
        rgb_features = self.rgb_stem(image)  # [B, C_f, H / 4, W / 4]
        if rgb_features.shape[-2:] != (target_height, target_width):
            rgb_features = nn.functional.interpolate(
                rgb_features,
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            )
        coarse_features = nn.functional.interpolate(
            self.coarse_projection(coarse_map),
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )  # [B, C_f, H / s, W / s]
        context_features = nn.functional.interpolate(
            self.context_projection(context_map),
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )  # [B, C_f, H / s, W / s]
        fused = self.fusion(
            torch.cat([rgb_features, coarse_features, context_features], dim=1)
        )  # [B, C_f, H / s, W / s]
        return F.normalize(fused + rgb_features, dim=1)


class TrackingHead(nn.Module):
    """
    Learned local-correlation tracker with visibility prediction.

    Given per-query fine features and a target feature map, computes a local
    correlation window around each current position estimate, applies
    soft-argmax to obtain a sub-pixel position delta, and predicts a
    visibility logit from correlation statistics.
    """

    def __init__(
        self,
        feature_dim: int,
        window_radius: int = 2,
        feature_stride: int = 4,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.window_radius = window_radius
        self.feature_stride = feature_stride
        self.temperature = temperature
        window_width = 2 * window_radius + 1
        window_area = window_width * window_width
        self.visibility_mlp = nn.Sequential(
            nn.Linear(window_area + 2, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        query_features: torch.Tensor,
        target_feature_map: torch.Tensor,
        current_positions: torch.Tensor,
    ):
        """
        Args:
            query_features:     [B, Q, C_f] reference-frame fine features per query.
            target_feature_map: [B, C_f, H_f, W_f] current-frame fine features.
            current_positions:  [B, Q, 2] current estimated pixel positions (x, y).

        Returns:
            position_delta:   [B, Q, 2] sub-pixel offset in image space.
            visibility_logit: [B, Q, 1] raw logit (apply sigmoid for probability).
            scores:           [B, Q] confidence from correlation peak.
        """
        B, Q, C = query_features.shape
        _, _, H_f, W_f = target_feature_map.shape
        device = query_features.device
        dtype = query_features.dtype

        feat_coords = (current_positions.float() + 0.5) / float(self.feature_stride) - 0.5  # [B, Q, 2]

        offsets = torch.arange(
            -self.window_radius, self.window_radius + 1, device=device, dtype=dtype
        )  # [W_w]
        off_y, off_x = torch.meshgrid(offsets, offsets, indexing="ij")
        offset_grid = torch.stack((off_x, off_y), dim=-1)  # [W_w, W_w, 2]

        window_coords = feat_coords.unsqueeze(2).unsqueeze(2) + offset_grid  # [B, Q, W_w, W_w, 2]

        grid = window_coords.clone()
        grid[..., 0] = grid[..., 0] / max(W_f - 1, 1) * 2 - 1
        grid[..., 1] = grid[..., 1] / max(H_f - 1, 1) * 2 - 1

        W_w = 2 * self.window_radius + 1
        grid_flat = grid.reshape(B * Q, W_w, W_w, 2)  # [B*Q, W_w, W_w, 2]
        tgt_expanded = target_feature_map.unsqueeze(1).expand(-1, Q, -1, -1, -1)  # [B, Q, C_f, H_f, W_f]
        tgt_flat = tgt_expanded.reshape(B * Q, C, H_f, W_f)  # [B*Q, C_f, H_f, W_f]
        windows = F.grid_sample(tgt_flat, grid_flat, mode="bilinear", align_corners=True)  # [B*Q, C_f, W_w, W_w]
        windows = F.normalize(windows, dim=1)

        queries_flat = F.normalize(query_features.reshape(B * Q, C), dim=1)  # [B*Q, C]
        logits = torch.einsum("mc,mcij->mij", queries_flat, windows)  # [B*Q, W_w, W_w]
        logits_flat = logits.reshape(B * Q, -1) / max(self.temperature, 1e-6)  # [B*Q, W_w*W_w]
        probs = torch.softmax(logits_flat, dim=1)  # [B*Q, W_w*W_w]
        probs_2d = probs.reshape(B * Q, W_w, W_w)  # [B*Q, W_w, W_w]

        offset_pixels = offset_grid * float(self.feature_stride)  # [W_w, W_w, 2]
        expected_offset = (probs_2d.unsqueeze(-1) * offset_pixels.unsqueeze(0)).sum(dim=(1, 2))  # [B*Q, 2]
        position_delta = expected_offset.reshape(B, Q, 2)  # [B, Q, 2]

        peak_prob = probs.max(dim=1).values  # [B*Q]
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)  # [B*Q]
        max_entropy = torch.log(torch.tensor(float(probs.shape[1]), device=device, dtype=dtype)).clamp_min(1e-6)
        entropy_conf = (1.0 - entropy / max_entropy).clamp(0.0, 1.0)  # [B*Q]
        scores = (0.5 * peak_prob + 0.5 * entropy_conf).clamp(0.0, 1.0).reshape(B, Q)

        delta_mag = position_delta.norm(dim=-1, keepdim=True).reshape(B * Q, 1)  # [B*Q, 1]
        vis_input = torch.cat([probs, peak_prob.unsqueeze(1), delta_mag], dim=1)  # [B*Q, W_w*W_w+2]
        visibility_logit = self.visibility_mlp(vis_input).reshape(B, Q, 1)  # [B, Q, 1]

        return position_delta, visibility_logit, scores


class MatcherModel(FeatureExtractor):
    def __init__(
        self,
        resampled_patch_size=8,
        shared_key=None,
        dino_model_name=None,
        dino_layers="auto",
        fusion_head="register_gated_hierarchical",
        fusion_gate_temperature=1.0,
        fusion_uniform_mixing=0.0,
        fusion_layer_dropout=0.1,
        fusion_gate_logit_bound=8.0,
        refinement_method="fft",
        fine_feature_dim=64,
        fine_feature_stride=4,
        refinement_context_layer="auto",
    ):
        super().__init__(
            shared_key=shared_key,
            dino_model_name=dino_model_name,
            image_size=384,
        )
        self.resampled_patch_size = resampled_patch_size
        self.fusion_head_name = fusion_head
        self.dino_layers = dino_layers
        self.latest_diagnostics = {}
        self.latest_refinement_feature_maps = {}
        self.refinement_method = str(refinement_method).lower()
        self.fine_feature_stride = max(1, int(fine_feature_stride))

        self.layer_indices = self._resolve_dinov3_layers(dino_layers)
        self.hierarchical_fusion = RegisterGatedHierarchicalFusion(
            hidden_dim=self.feature_dim,
            layer_indices=self.layer_indices,
            num_register_tokens=self.num_register_tokens,
            gate_temperature=fusion_gate_temperature,
            uniform_mixing=fusion_uniform_mixing,
            layer_dropout_p=fusion_layer_dropout,
            gate_logit_bound=fusion_gate_logit_bound,
        )
        self.refinement_context_layer = self._resolve_refinement_context_layer(
            refinement_context_layer
        )

        self.fine_feature_head = None
        if self.refinement_method == "feature_softargmax":
            self.fine_feature_head = LocalRefinementFeatureHead(
                image_channels=3,
                descriptor_dim=self.feature_dim,
                output_dim=int(fine_feature_dim),
                feature_stride=self.fine_feature_stride,
            )

        self.tracking_head = None
        self._fine_feature_dim = int(fine_feature_dim)

        self.patchsize_resampler = lambda x: nn.functional.interpolate(
            x,
            size=(384 // self.resampled_patch_size, 384 // self.resampled_patch_size),
            mode="bilinear",
            align_corners=False,
        )

    def _resolve_dinov3_layers(self, dino_layers):
        num_layers = getattr(self.backbone, "num_hidden_layers", 12)
        if dino_layers is None or str(dino_layers).lower() == "auto":
            n_selected = min(4, num_layers)
            return sorted(
                {
                    int(layer_idx)
                    for layer_idx in torch.linspace(
                        1, num_layers, steps=n_selected
                    ).round().tolist()
                }
            )
        if isinstance(dino_layers, str):
            if dino_layers.lower() == "all":
                return list(range(1, num_layers + 1))
            return [
                max(1, min(num_layers, int(layer.strip())))
                for layer in dino_layers.split(",")
                if layer.strip()
            ]
        return [max(1, min(num_layers, int(layer))) for layer in dino_layers]

    def _resolve_refinement_context_layer(self, refinement_context_layer):
        num_layers = getattr(self.backbone, "num_hidden_layers", 12)
        if (
            refinement_context_layer is None
            or str(refinement_context_layer).lower() == "auto"
        ):
            if self.layer_indices:
                return min(self.layer_indices)
            return max(1, min(num_layers, 2))
        return max(1, min(num_layers, int(refinement_context_layer)))

    def _encode_image(self, image):
        with torch.no_grad():
            features = self.extract_features(image)

        raw_patch_tokens = self.extract_patch_tokens(features[-1])  # [B, N, C]
        matched_patch_tokens, diagnostics = self.hierarchical_fusion(
            features, return_diagnostics=True
        )  # [B, N, C]
        gated_layer_maps = build_gated_layer_maps(diagnostics)
        refinement_context_tokens = self.extract_patch_tokens(
            features[self.refinement_context_layer]
        )  # [B, N, C]
        cls_token = self.extract_cls_token(features[-1])  # [B, C]
        fine_feature_map = self._build_refinement_feature_map(
            image=image,
            matched_patch_tokens=matched_patch_tokens,
            refinement_context_tokens=refinement_context_tokens,
        )
        return (
            raw_patch_tokens,
            matched_patch_tokens,
            cls_token,
            diagnostics,
            fine_feature_map,
            gated_layer_maps,
        )

    def _build_refinement_feature_map(
        self,
        image: torch.Tensor,
        matched_patch_tokens: torch.Tensor,
        refinement_context_tokens: torch.Tensor,
    ):
        if self.fine_feature_head is None:
            return None
        return self.fine_feature_head(
            image=image,
            coarse_tokens=matched_patch_tokens.permute(0, 2, 1),
            context_tokens=refinement_context_tokens.permute(0, 2, 1),
        )

    def enable_tracking_head(self, window_radius: int = 2, temperature: float = 0.1):
        """Instantiate the tracking head (idempotent)."""
        if self.tracking_head is not None:
            return
        device = next(self.parameters()).device
        self.tracking_head = TrackingHead(
            feature_dim=self._fine_feature_dim,
            window_radius=window_radius,
            feature_stride=self.fine_feature_stride,
            temperature=temperature,
        ).to(device)

    def save_pretrained_descriptors(self, path: str) -> str:
        """Save only the descriptor-relevant weights (fusion head + fine feature head).

        This produces a lightweight checkpoint suitable for loading into Phase 2
        (tracking) with frozen descriptor layers. Tracking head weights are
        stripped so the checkpoint stays small and portable.

        Args:
            path: Destination ``.pt`` file path.

        Returns:
            The *path* written.
        """
        exclude_prefixes = ("tracking_head.",)
        descriptor_state = {
            k: v for k, v in self.state_dict().items()
            if not k.startswith(exclude_prefixes)
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"model_state_dict": descriptor_state}, path)
        logger.info(f"Saved pretrained descriptor checkpoint ({len(descriptor_state)} keys) → {path}")
        return path

    def _load_checkpoint_state_dict(self, state_dict):
        # Filter out legacy DPT keys that may exist in old checkpoints
        legacy_prefixes = ("depthpredictor.", "depth_backbone.")
        state_dict = {
            k: v for k, v in state_dict.items()
            if not k.startswith(legacy_prefixes)
        }

        # Checkpoints from PHASE=end2end may include TrackingHead; load without error when
        # this matcher has no `tracking_head` (e.g. PHASE=tracking only builds refinement).
        if getattr(self, "tracking_head", None) is None:
            state_dict = {
                k: v for k, v in state_dict.items()
                if not k.startswith("tracking_head.")
            }

        if self.fine_feature_head is None and self.tracking_head is None:
            self.load_state_dict(state_dict, strict=False)
            return

        incompatible = self.load_state_dict(state_dict, strict=False)
        allowed_missing_prefixes = ("fine_feature_head.", "tracking_head.")
        disallowed_missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith(allowed_missing_prefixes)
        ]
        if disallowed_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Checkpoint is incompatible with the current matcher. "
                f"Missing keys: {disallowed_missing}, unexpected keys: {incompatible.unexpected_keys}"
            )
        if incompatible.missing_keys:
            logger.warning(
                "Checkpoint loaded without fine refinement / tracking head weights; "
                "new branches will start from random initialization."
            )

    def forward(self, framestack):
        """
        Args:
            framestack: torch.Tensor, shape [B, 2, 3, H, W]
        Returns:
            dict with source/target embeddings for matching
        """
        source = framestack[:, 0]  # [B, 3, H, W]
        target = framestack[:, -1]  # [B, 3, H, W]
        (
            source_raw_patch_tokens,
            source_matched_patch_tokens,
            source_cls_token,
            source_diagnostics,
            source_fine_feature_map,
            source_gated_maps,
        ) = self._encode_image(source)
        (
            target_raw_patch_tokens,
            target_matched_patch_tokens,
            target_cls_token,
            target_diagnostics,
            target_fine_feature_map,
            target_gated_maps,
        ) = self._encode_image(target)
        self.latest_diagnostics = {
            "source": source_diagnostics,
            "target": target_diagnostics,
        }
        self.latest_refinement_feature_maps = {
            "source": source_fine_feature_map,
            "target": target_fine_feature_map,
        }
        self.latest_gated_layer_maps = {
            "source": source_gated_maps,
            "target": target_gated_maps,
        }

        mono3doutput = {
            "source_embedding": source_raw_patch_tokens.permute(0, 2, 1),  # [B, C, HW]
            "target_embedding": target_raw_patch_tokens.permute(0, 2, 1),  # [B, C, HW]
            "source_embedding_match": source_matched_patch_tokens.permute(0, 2, 1),  # [B, C, HW]
            "target_embedding_match": target_matched_patch_tokens.permute(0, 2, 1),  # [B, C, HW]
            "source_cls": source_cls_token,  # [B, C]
            "target_cls": target_cls_token,  # [B, C]
        }

        if self.resampled_patch_size != 16:
            for key in [
                "source_embedding",
                "target_embedding",
                "source_embedding_match",
                "target_embedding_match",
            ]:
                mono3doutput[key] = chw2embedding(
                    self.patchsize_resampler(embedding2chw(mono3doutput[key], False))
                )

        return mono3doutput

    def embed(self, *frames, mode="chw", loftr_shape=False):
        """Inference-only method to return embeddings for one or more frames."""
        assert mode in ["chw", "seq"], "Invalid mode. Must be 'chw' or 'seq'."
        embs = []
        for image in frames:
            _, matched_patch_tokens, _, _, _, _ = self._encode_image(image)
            emb = embedding2chw(matched_patch_tokens)  # [B, C, H_p, W_p]
            if loftr_shape or self.resampled_patch_size != 16:
                emb = self.patchsize_resampler(emb)
            embs.append(emb)
        embs = [chw2embedding(emb) if mode == "seq" else emb for emb in embs]
        if len(embs) == 1:
            return embs[0]
        return embs

    def fromArtifact(
        self,
        model_name=None,
        local_path=None,
        bucket="alberto-bucket",
        device="cuda",
        pth_namestring=None,
    ):
        """
        Loads a PyTorch model from Google Cloud Storage or local checkpoint

        Args:
            bucket (str): Name of the GCS bucket
            model_name (str): Path to the model file in the bucket
            device (str): Device to load the model on ('cuda' or 'cpu')

        Returns:
            torch.nn.Module: Loaded model, or None if loading failed
        """
        if pth_namestring is not None:
            if not os.path.exists(pth_namestring):
                raise FileNotFoundError(
                    f"Model checkpoint not found at {pth_namestring}"
                )
            loaded_dict = _load_checkpoint_mapping(pth_namestring, map_location=device)
            if "model_state_dict" in loaded_dict.keys():
                self._load_checkpoint_state_dict(loaded_dict["model_state_dict"])
            else:
                self._load_checkpoint_state_dict(loaded_dict)
            return
        if local_path is not None:
            pth_namestring = os.path.join(local_path, f"{model_name}_checkpoint.pth")
            if not os.path.exists(pth_namestring):
                raise FileNotFoundError(
                    f"Model checkpoint not found at {pth_namestring}"
                )
            loaded_dict = _load_checkpoint_mapping(pth_namestring, map_location=device)
            self._load_checkpoint_state_dict(loaded_dict["model_state_dict"])

            if os.path.dirname(pth_namestring) == tempfile.gettempdir():
                os.remove(local_path)
        else:
            try:
                local_path = download_from_gcs(model_name=model_name, bucket_name=bucket)
            except Exception:
                local_path = None

            if local_path is None:
                possible_paths = [
                    os.path.join("runs", model_name, "checkpoints", "weights_best.pt"),
                    os.path.join("checkpoints", f"{model_name}_checkpoint.pt"),
                    os.path.join("checkpoints", f"{model_name}.pt"),
                    os.path.join("checkpoints", "weights_best.pt"),
                ]

                local_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        local_path = path
                        break

                if local_path is None:
                    raise FileNotFoundError(
                        f"Model '{model_name}' not found in GCS bucket {bucket} or locally. "
                        f"Checked paths: {possible_paths}"
                    )
                else:
                    loaded_dict = _load_checkpoint_mapping(local_path, map_location=device)
                    self._load_checkpoint_state_dict(loaded_dict["model_state_dict"])
