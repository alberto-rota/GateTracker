import os
import tempfile

import torch
import torch.nn as nn

from gatetracker.backbone.dinov3 import DINOv3
from gatetracker.utils.logger import get_logger
from utilities.dev_utils import download_from_gcs
from gatetracker.utils.tensor_ops import chw2embedding, embedding2chw

_SHARED_EXTRACTORS = {}
logger = get_logger(__name__)


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        shared_key=None,
        backbone_family="dinov3",
        dino_model_name=None,
        image_size=384,
    ):
        super().__init__()
        self.shared_key = shared_key
        self.backbone_family = str(backbone_family).lower()
        self.dino_model_name = dino_model_name
        self.image_size = image_size

        if shared_key is not None and shared_key in _SHARED_EXTRACTORS:
            self._copy_shared_components(_SHARED_EXTRACTORS[shared_key])
            self._is_shared = True
        else:
            self._build_backbone_components()
            self._is_shared = False
            if shared_key is not None:
                _SHARED_EXTRACTORS[shared_key] = self

    @property
    def device(self):
        return next(self.parameters()).device

    def parameters_summary(self, verbose=False):
        for name, parameter in self.named_parameters():
            if verbose:
                print(f"{name} : {parameter.numel()}")
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        untrainable = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"TRAINABLE Parameters: {trainable}")
        print(f"UNTRAINABLE Parameters: {untrainable}")
        print(f"TOTAL Parameters: {total}")
        return {"trainable": trainable, "untrainable": untrainable, "total": total}

    def _copy_shared_components(self, shared_extractor):
        for attr in [
            "backbone",
            "lastvitlayer",
            "feature_dim",
            "patch_size",
            "num_prefix_tokens",
            "num_register_tokens",
            "backbone_output_key",
        ]:
            setattr(self, attr, getattr(shared_extractor, attr))

    def _build_backbone_components(self):
        self.backbone = DINOv3(
            {
                "model_name": self.dino_model_name
                or "facebook/dinov3-vitb16-pretrain-lvd1689m",
                "image_size": self.image_size,
                "freeze_backbone": True,
                "return_last_hidden_state": True,
                "return_all_hidden_states": True,
                "return_patch_tokens_only": False,
                "return_cls_token": True,
                "return_register_tokens": True,
            }
        )
        self.lastvitlayer = nn.Identity()
        self.feature_dim = self.backbone.feature_dim
        self.patch_size = self.backbone.patch_size
        self.num_register_tokens = self.backbone.num_register_tokens
        self.num_prefix_tokens = 1 + self.num_register_tokens
        self.backbone_output_key = "hidden_states"

    def extract_features(self, framestack):
        """
        Args:
            framestack: [B, 3, H, W] or [B, S, 3, H, W]
        Returns:
            list of per-layer features
            - token backbones: [B, N_tokens, C]
            - sequence mode: [B, S, N_tokens, C]
        """
        is_sequence = framestack.ndim == 5
        if is_sequence:
            batch_size, seq_len = framestack.shape[:2]
            framestack = framestack.reshape(-1, *framestack.shape[2:])  # [B*S, 3, H, W]

        features = self.backbone(framestack)[self.backbone_output_key]
        if isinstance(features, tuple):
            features = list(features)
        elif isinstance(features, torch.Tensor):
            features = [features]

        if is_sequence:
            features = [
                feature.reshape(batch_size, seq_len, *feature.shape[1:])
                for feature in features
            ]

        return features

    def extract_cls_token(self, hidden_states):
        return hidden_states[..., 0, :]

    def extract_register_tokens(self, hidden_states):
        if self.num_register_tokens == 0:
            return None
        return hidden_states[..., 1 : 1 + self.num_register_tokens, :]

    def extract_patch_tokens(self, hidden_states):
        return hidden_states[..., self.num_prefix_tokens :, :]

    def tokens_to_feature_maps(self, patch_tokens, batch_size, patch_h, patch_w):
        patch_tokens = patch_tokens.transpose(1, 2).contiguous()  # [B, C, N_patches]
        return patch_tokens.view(batch_size, patch_tokens.shape[1], patch_h, patch_w)  # [B, C, pH, pW]

    def get_patch_spatial_dims(self, input_height, input_width):
        return input_height // self.patch_size, input_width // self.patch_size

    def embed(self, *frames, mode="chw", loftr_shape=False):
        assert mode in ["chw", "seq"], "Invalid mode. Must be 'chw' or 'seq'."
        embeddings = [
            self._extract_embeddings(self.extract_features(image), loftr_shape)
            for image in frames
        ]
        embeddings = [
            chw2embedding(embedding) if mode == "seq" else embedding
            for embedding in embeddings
        ]
        return embeddings[0] if len(embeddings) == 1 else embeddings

    def fromArtifact(
        self,
        model_name,
        bucket="alberto-bucket",
        device="cuda",
    ):
        """
        Loads a PyTorch model from Google Cloud Storage or local checkpoint.
        """
        try:
            local_path = download_from_gcs(model_name=model_name, bucket_name=bucket)
        except Exception as exc:
            logger.warning(f"Failed to download from GCS: {str(exc)}")
            local_path = None

        if local_path is None:
            local_path = os.path.join("checkpoints", "weights_best.pt")
            if not os.path.exists(local_path):
                raise FileNotFoundError(
                    f"Model not found in GCS bucket {bucket} or locally at {local_path}"
                )

        loaded_dict = torch.load(local_path, map_location=device, weights_only=True)
        self.load_state_dict(loaded_dict)

        if os.path.dirname(local_path) == tempfile.gettempdir():
            os.remove(local_path)
