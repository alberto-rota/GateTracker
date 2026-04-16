import copy
import os
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

import networks.backbones as backbones
from networks.base import MONO3DModel
from logger import get_logger
from utilities.dev_utils import download_from_gcs
from utilities.tensor_utils import chw2embedding, embedding2chw

# Global registry for sharing frozen backbone extractors
_SHARED_EXTRACTORS = {}
logger = get_logger(__name__)


class FeatureExtractor(MONO3DModel):
    def __init__(
        self,
        backbone_brand="intel",
        size="beit-base-384",
        shared_key=None,
        backbone_family="dinov2",
        dino_model_name=None,
        image_size=384,
    ):
        super(FeatureExtractor, self).__init__()
        self.backbone_brand = backbone_brand
        self.size = size
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
    def is_dinov3(self):
        return self.backbone_family == "dinov3"

    def _copy_shared_components(self, shared_extractor):
        for attr in [
            "backbone",
            "backbone_out_indices",
            "lastvitlayer",
            "feature_dim",
            "patch_size",
            "num_prefix_tokens",
            "num_register_tokens",
            "backbone_output_key",
        ]:
            setattr(self, attr, getattr(shared_extractor, attr))

    def _build_backbone_components(self):
        if self.is_dinov3:
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
            self.backbone_out_indices = None
            self.lastvitlayer = nn.Identity()
            self.feature_dim = self.backbone.feature_dim
            self.patch_size = self.backbone.patch_size
            self.num_register_tokens = self.backbone.num_register_tokens
            self.num_prefix_tokens = 1 + self.num_register_tokens
            self.backbone_output_key = "hidden_states"
            return

        self.backbone = getattr(
            backbones, f"DINOv2_{self.backbone_brand.capitalize()}"
        )(size=self.size)
        config_dict = self.backbone.model.config.to_dict()
        output_index_key = (
            "out_indices"
            if "swin" in self.size
            or "beit" in self.size
            or "facebook" in self.backbone_brand
            else "backbone_out_indices"
        )
        self.backbone_out_indices = config_dict[output_index_key]
        self.lastvitlayer = copy.deepcopy(self.backbone.model.encoder.layer[-1])
        for param in self.lastvitlayer.parameters():
            param.requires_grad = True
        self.feature_dim = config_dict.get(
            "hidden_size",
            self.backbone.get_feature_dim() if hasattr(self.backbone, "get_feature_dim") else None,
        )
        self.patch_size = config_dict.get("patch_size", 16)
        self.num_register_tokens = 0
        self.num_prefix_tokens = 1
        self.backbone_output_key = (
            "feature_maps"
            if "swin" in self.size or "beit" in self.size
            else "hidden_states"
        )

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
            framestack = framestack.reshape(-1, *framestack.shape[2:])

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
        patch_tokens = patch_tokens.transpose(1, 2).contiguous()
        return patch_tokens.view(batch_size, patch_tokens.shape[1], patch_h, patch_w)

    def get_patch_spatial_dims(self, input_height, input_width):
        return input_height // self.patch_size, input_width // self.patch_size

    def _compute_depth(self, source_features_dino):
        depthstack = self.depthpredictor(
            source_features_dino,
            self.backbone_out_indices,
        )
        return depthstack.view(-1, 1, depthstack.shape[-2], depthstack.shape[-1])

    def _extract_embeddings(self, features, loftr_shape=True):
        patch_tokens = self.extract_patch_tokens(features[-1])
        if loftr_shape:
            return self.patchsize_resampler(embedding2chw(patch_tokens))
        return embedding2chw(patch_tokens)

    def depth(self, framestack):
        source = framestack[:, 0] if framestack.ndim == 5 else framestack
        return self._compute_depth(self.extract_features(source))

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


class DINOv3(nn.Module):
    """
    Hugging Face DINOv3 wrapper with vectorized preprocessing and flexible
    hidden-state extraction.
    """

    def __init__(self, config):
        super().__init__()
        self.config = {
            "model_name": "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "image_size": 384,
            "freeze_backbone": True,
            "return_last_hidden_state": True,
            "return_all_hidden_states": False,
            "return_selected_layers": None,
            "return_patch_tokens_only": False,
            "return_as_feature_maps": False,
            "return_cls_token": False,
            "return_register_tokens": False,
            **config,
        }

        self.dinov3 = AutoModel.from_pretrained(self.config["model_name"])
        self.processor = AutoImageProcessor.from_pretrained(self.config["model_name"])
        self.feature_dim = self.dinov3.config.hidden_size
        self.patch_size = getattr(self.dinov3.config, "patch_size", 16)
        self.num_hidden_layers = getattr(self.dinov3.config, "num_hidden_layers", 12)
        self.num_register_tokens = getattr(
            self.dinov3.config, "num_register_tokens", 4
        )
        self.num_prefix_tokens = 1 + self.num_register_tokens

        image_mean = torch.tensor(
            getattr(self.processor, "image_mean", [0.485, 0.456, 0.406]),
            dtype=torch.float32,
        )
        image_std = torch.tensor(
            getattr(self.processor, "image_std", [0.229, 0.224, 0.225]),
            dtype=torch.float32,
        )
        if image_mean.numel() == 1:
            image_mean = image_mean.repeat(3)
            image_std = image_std.repeat(3)
        self.register_buffer(
            "pixel_mean", image_mean.view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "pixel_std", image_std.view(1, 3, 1, 1), persistent=False
        )

        if self.config["freeze_backbone"]:
            for param in self.dinov3.parameters():
                param.requires_grad = False

    def preprocess_image(self, image_tensor):
        """
        Args:
            image_tensor: [B, 3, H, W], values in [0, 1] or [0, 255]
        Returns:
            [B, 3, H, W] normalized for DINOv3
        """
        image_tensor = image_tensor.float()
        if image_tensor.max() > 1.5:
            image_tensor = image_tensor / 255.0
        image_tensor = image_tensor.clamp(0.0, 1.0)

        target_size = self.config.get("image_size")
        if target_size is not None and image_tensor.shape[-2:] != (target_size, target_size):
            image_tensor = F.interpolate(
                image_tensor,
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            )

        return (image_tensor - self.pixel_mean) / self.pixel_std

    def extract_cls_token(self, hidden_states):
        return hidden_states[:, 0]

    def extract_register_tokens(self, hidden_states):
        return hidden_states[:, 1 : 1 + self.num_register_tokens]

    def extract_patch_tokens(self, hidden_states):
        return hidden_states[:, self.num_prefix_tokens :]

    def patch_tokens_to_feature_maps(self, patch_tokens, batch_size, patch_h, patch_w):
        patch_tokens = patch_tokens.transpose(1, 2).contiguous()
        return patch_tokens.view(batch_size, patch_tokens.shape[1], patch_h, patch_w)

    def get_patch_spatial_dims(self, input_height, input_width):
        return input_height // self.patch_size, input_width // self.patch_size

    def forward(self, rgb_image):
        pixel_values = self.preprocess_image(rgb_image)
        batch_size, _, input_h, input_w = pixel_values.shape
        assert input_h % self.patch_size == 0, (
            f"Height {input_h} must be divisible by patch size {self.patch_size}"
        )
        assert input_w % self.patch_size == 0, (
            f"Width {input_w} must be divisible by patch size {self.patch_size}"
        )

        patch_h, patch_w = self.get_patch_spatial_dims(input_h, input_w)
        need_all_hidden_states = (
            self.config["return_all_hidden_states"]
            or self.config["return_selected_layers"] is not None
        )
        outputs = self.dinov3(pixel_values, output_hidden_states=need_all_hidden_states)

        result = {}
        last_hidden = outputs.last_hidden_state

        if self.config["return_last_hidden_state"]:
            result["last_hidden_state"] = (
                self.patch_tokens_to_feature_maps(
                    self.extract_patch_tokens(last_hidden), batch_size, patch_h, patch_w
                )
                if self.config["return_as_feature_maps"]
                else last_hidden
            )

        if self.config["return_cls_token"]:
            result["cls_token"] = self.extract_cls_token(last_hidden)

        if self.config["return_register_tokens"]:
            result["register_tokens"] = self.extract_register_tokens(last_hidden)

        if self.config["return_all_hidden_states"]:
            all_hidden = list(outputs.hidden_states)
            if self.config["return_patch_tokens_only"]:
                all_hidden = [self.extract_patch_tokens(hidden) for hidden in all_hidden]
            if self.config["return_as_feature_maps"]:
                all_hidden = [
                    self.patch_tokens_to_feature_maps(
                        hidden
                        if self.config["return_patch_tokens_only"]
                        else self.extract_patch_tokens(hidden),
                        batch_size,
                        patch_h,
                        patch_w,
                    )
                    for hidden in all_hidden
                ]
            result["all_hidden_states"] = all_hidden
            result["hidden_states"] = all_hidden

        if self.config["return_selected_layers"] is not None:
            selected_hidden = [outputs.hidden_states[i] for i in self.config["return_selected_layers"]]
            if self.config["return_patch_tokens_only"]:
                selected_hidden = [
                    self.extract_patch_tokens(hidden) for hidden in selected_hidden
                ]
            if self.config["return_as_feature_maps"]:
                selected_hidden = [
                    self.patch_tokens_to_feature_maps(
                        hidden
                        if self.config["return_patch_tokens_only"]
                        else self.extract_patch_tokens(hidden),
                        batch_size,
                        patch_h,
                        patch_w,
                    )
                    for hidden in selected_hidden
                ]
            result["selected_hidden_states"] = selected_hidden

        return result