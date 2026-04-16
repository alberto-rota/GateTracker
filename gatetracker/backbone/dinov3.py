import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel


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
