"""Tests for pipeline phase selection and optimizer param groups."""

import pytest
import torch
import torch.nn as nn
from dotmap import DotMap

import gatetracker.backbone.feature_extractor as featureextractor_module
from gatetracker.matching.model import MatcherModel


HEIGHT = WIDTH = 384
HIDDEN_DIM = 32
PATCH_SIZE = 16


class FakeDINOv3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feature_dim = HIDDEN_DIM
        self.patch_size = PATCH_SIZE
        self.num_hidden_layers = 12
        self.num_register_tokens = 4
        self.patch_embed = nn.Conv2d(
            3, HIDDEN_DIM, kernel_size=PATCH_SIZE, stride=PATCH_SIZE, bias=False
        )

    def forward(self, image):
        patch_tokens = self.patch_embed(image).flatten(2).transpose(1, 2)
        cls_token = patch_tokens.mean(dim=1, keepdim=True)
        register_tokens = patch_tokens[:, :1, :].repeat(1, self.num_register_tokens, 1)
        current = torch.cat([cls_token, register_tokens, patch_tokens], dim=1)
        hidden_states = [current]
        for layer_idx in range(self.num_hidden_layers):
            current = current + (layer_idx + 1) * 0.01
            hidden_states.append(current)
        return {
            "last_hidden_state": hidden_states[-1],
            "hidden_states": hidden_states,
            "all_hidden_states": hidden_states,
            "cls_token": hidden_states[-1][:, 0, :],
            "register_tokens": hidden_states[-1][:, 1 : 1 + self.num_register_tokens, :],
        }


@pytest.fixture(autouse=True)
def patch_backbones(monkeypatch):
    featureextractor_module._SHARED_EXTRACTORS.clear()
    monkeypatch.setattr(featureextractor_module, "DINOv3", FakeDINOv3)


from gatetracker.utils.training_phase import (
    build_optimizer_param_groups,
    matcher_should_enable_tracking_head,
    normalize_pipeline_phase,
    pairwise_tracking_enabled,
)


def test_normalize_pipeline_phase():
    assert normalize_pipeline_phase(DotMap({"PHASE": "pretrain"})) == "pretrain"
    assert normalize_pipeline_phase(DotMap({"PHASE": "END2END"})) == "end2end"
    assert normalize_pipeline_phase(DotMap({"PHASE": "tracking"})) == "tracking"


def test_pairwise_tracking_enabled():
    assert pairwise_tracking_enabled(DotMap({"PHASE": "pretrain", "TRACKING_MODE": False})) is False
    assert pairwise_tracking_enabled(DotMap({"PHASE": "end2end"})) is True
    assert pairwise_tracking_enabled(DotMap({"PHASE": "tracking"})) is False
    assert pairwise_tracking_enabled(DotMap({"PHASE": "pretrain", "TRACKING_MODE": True})) is True


def test_matcher_should_enable_tracking_head():
    assert not matcher_should_enable_tracking_head(
        DotMap({"PHASE": "pretrain", "TRACKING_MODE": False, "TRACKING_HEAD": False})
    )
    assert matcher_should_enable_tracking_head(
        DotMap({"PHASE": "pretrain", "TRACKING_HEAD": True})
    )
    assert matcher_should_enable_tracking_head(DotMap({"PHASE": "end2end"}))


def test_build_optimizer_param_groups_splits_modules():
    model = MatcherModel(
        resampled_patch_size=16,
        dino_model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        refinement_method="feature_softargmax",
    )
    model.enable_tracking_head()
    cfg = DotMap(
        {
            "LEARNING_RATE": 1e-4,
            "LR_FUSION": 1e-5,
            "LR_FINE_FEATURE": 2e-5,
            "LR_TRACKING_HEAD": 3e-5,
        }
    )
    groups = build_optimizer_param_groups(model, cfg)
    names = [g["group_name"] for g in groups]
    assert "hierarchical_fusion" in names
    assert "fine_feature_head" in names
    assert "tracking_head" in names
    assert groups[names.index("hierarchical_fusion")]["lr"] == 1e-5
    assert groups[names.index("fine_feature_head")]["lr"] == 2e-5
    assert groups[names.index("tracking_head")]["lr"] == 3e-5


def test_build_optimizer_param_groups_empty_trainable():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2, bias=False)

    m = Tiny()
    for p in m.parameters():
        p.requires_grad = False
    groups = build_optimizer_param_groups(m, DotMap({"LEARNING_RATE": 0.01}))
    assert len(groups) == 1
    assert groups[0]["group_name"] == "all"
