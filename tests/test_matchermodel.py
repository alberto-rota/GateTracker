import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotmap import DotMap

import gatetracker.backbone.feature_extractor as featureextractor_module
import gatetracker.matching.metrics as matching_metrics_module
import gatetracker.matching.model as model_module
import gatetracker.matching.refinement as refinement_module
from gatetracker.losses import InfoNCELoss, WeightedSmoothL1Loss

HEIGHT = 384
WIDTH = 384
HIDDEN_DIM = 32
PATCH_SIZE = 16
GRID_SIZE = HEIGHT // PATCH_SIZE
SEQ_LEN = GRID_SIZE * GRID_SIZE


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
        patch_tokens = self.patch_embed(image).flatten(2).transpose(1, 2)  # [B, N, C]
        cls_token = patch_tokens.mean(dim=1, keepdim=True)  # [B, 1, C]
        register_tokens = patch_tokens[:, :1, :].repeat(1, self.num_register_tokens, 1)  # [B, 4, C]
        hidden_states = []
        current = torch.cat([cls_token, register_tokens, patch_tokens], dim=1)  # [B, 1+4+N, C]
        hidden_states.append(current)
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


@pytest.fixture
def framestack():
    return torch.rand(2, 2, 3, HEIGHT, WIDTH)


@pytest.fixture
def matcher_model():
    return model_module.MatcherModel(
        resampled_patch_size=16,
        dino_model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        dino_layers="auto",
        fusion_head="register_gated_hierarchical",
    )


def test_dinov3_matcher_outputs_preserve_api(matcher_model, framestack):
    matcher_model.eval()
    model_output = matcher_model(framestack)

    expected_keys = {
        "source_embedding",
        "target_embedding",
        "source_embedding_match",
        "target_embedding_match",
        "source_cls",
        "target_cls",
    }
    assert expected_keys == set(model_output.keys())
    assert model_output["source_embedding"].shape == (2, HIDDEN_DIM, SEQ_LEN)
    assert model_output["target_embedding"].shape == (2, HIDDEN_DIM, SEQ_LEN)
    assert model_output["source_embedding_match"].shape == (2, HIDDEN_DIM, SEQ_LEN)
    assert model_output["target_embedding_match"].shape == (2, HIDDEN_DIM, SEQ_LEN)
    assert model_output["source_cls"].shape == (2, HIDDEN_DIM)
    assert model_output["target_cls"].shape == (2, HIDDEN_DIM)
    assert torch.isfinite(model_output["source_embedding_match"]).all()

    diagnostics = matcher_model.latest_diagnostics
    assert diagnostics["source"] is not None
    assert diagnostics["target"] is not None
    assert diagnostics["source"]["layer_weights"].shape[:2] == (2, 4)
    assert diagnostics["target"]["layer_weights"].shape[:2] == (2, 4)
    assert diagnostics["source"]["layer_weights"].max() < 0.9
    assert diagnostics["source"]["layer_weights"].min() > 0.0

    embedded = matcher_model.embed(framestack[:, 0], mode="seq")
    assert embedded.shape == (2, HIDDEN_DIM, SEQ_LEN)


def test_dinov3_matcher_backward_updates_hierarchical_fusion(matcher_model, framestack):
    matcher_model.train()
    model_output = matcher_model(framestack)
    loss = (
        model_output["source_embedding_match"].mean()
        + model_output["target_embedding_match"].mean()
        + model_output["source_cls"].mean()
        + model_output["target_cls"].mean()
    )
    matcher_model.zero_grad()
    loss.backward()

    grads = [
        param.grad
        for param in matcher_model.hierarchical_fusion.parameters()
        if param.requires_grad
    ]
    assert grads, "Hierarchical fusion head should expose trainable parameters."
    assert any(
        grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
        for grad in grads
    ), "Hierarchical fusion head did not receive a valid gradient."


def test_dinov3_feature_refiner_populates_internal_feature_maps(framestack):
    matcher_model = model_module.MatcherModel(
        resampled_patch_size=16,
        dino_model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        dino_layers="auto",
        fusion_head="register_gated_hierarchical",
        refinement_method="feature_softargmax",
        fine_feature_dim=24,
        fine_feature_stride=4,
    )
    matcher_model.eval()
    model_output = matcher_model(framestack)

    assert set(model_output.keys()) == {
        "source_embedding",
        "target_embedding",
        "source_embedding_match",
        "target_embedding_match",
        "source_cls",
        "target_cls",
    }
    fine_maps = matcher_model.latest_refinement_feature_maps
    assert fine_maps["source"].shape == (2, 24, HEIGHT // 4, WIDTH // 4)
    assert fine_maps["target"].shape == (2, 24, HEIGHT // 4, WIDTH // 4)
    assert torch.isfinite(fine_maps["source"]).all()
    assert torch.isfinite(fine_maps["target"]).all()


def test_feature_softargmax_refiner_recovers_local_shift():
    descriptor = F.normalize(torch.tensor([1.0, 2.0, 3.0, 4.0]), dim=0)
    source_map = (-descriptor).view(1, 4, 1, 1).repeat(1, 1, 9, 9)
    target_map = source_map.clone()
    source_map[0, :, 4, 4] = descriptor
    target_map[0, :, 3, 5] = descriptor

    src_xy, tgt_xy, scores = refinement_module.feature_softargmax_refiner(
        source_feature_map=source_map,
        target_feature_map=target_map,
        source_pixels=torch.tensor([[4.0, 4.0]]),
        target_pixels=torch.tensor([[4.0, 4.0]]),
        batch_indices=torch.tensor([0], dtype=torch.long),
        window_radius=2,
        feature_stride=1,
        softmax_temperature=0.05,
    )

    assert torch.allclose(src_xy, torch.tensor([[2.0, 2.0]]), atol=1e-4)
    assert torch.allclose(tgt_xy, torch.tensor([[3.0, 1.0]]), atol=0.2)
    assert scores.item() > 0.9


def test_refinement_metrics_report_pre_post_gain():
    class DummyMatcher:
        pass

    pipeline = DummyMatcher()
    pipeline.latest_refinement_state = {
        "scores": torch.tensor([0.9, 0.1], dtype=torch.float32),
        "active_mask": torch.tensor([True, False]),
        "coarse_target_pixels": torch.tensor(
            [[0.0, 0.0], [10.0, 0.0]], dtype=torch.float32
        ),
        "refined_target_pixels": torch.tensor(
            [[1.0, 0.0], [11.0, 0.0]], dtype=torch.float32
        ),
    }
    metrics = matching_metrics_module.refinement_metrics(
        pipeline,
        true_pixels_matched=torch.tensor(
            [[1.0, 0.0], [12.0, 0.0]], dtype=torch.float32
        ),
    )

    assert metrics["RefinementActiveFraction"] == pytest.approx(0.5)
    assert metrics["RefinementOffsetMean"] == pytest.approx(1.0)
    assert metrics["RefinementScoreMean"] == pytest.approx(0.5)
    assert metrics["CoarseErrorMean"] == pytest.approx(1.5)
    assert metrics["RefinedErrorMean"] == pytest.approx(0.5)
    assert metrics["RefinementGainPx"] == pytest.approx(1.0)
    assert metrics["RefinementWinRate"] == pytest.approx(1.0)


def test_infonce_respects_confidence_weights():
    loss_fn = InfoNCELoss(temperature=0.1, symmetric=False)
    logits = torch.tensor(
        [
            [5.0, 0.0],
            [5.0, 0.0],
        ]
    )
    positives = torch.tensor([0, 1], dtype=torch.long)
    mask = torch.ones_like(logits, dtype=torch.bool)

    weighted_loss = loss_fn(
        source_to_target_logits=logits,
        source_to_target_positive=positives,
        source_to_target_mask=mask,
        contrastive_weights=torch.tensor([1.0, 0.0]),
    )
    unweighted_loss = loss_fn(
        source_to_target_logits=logits,
        source_to_target_positive=positives,
        source_to_target_mask=mask,
        contrastive_weights=torch.tensor([1.0, 1.0]),
    )
    assert weighted_loss < unweighted_loss
