import pytest
import torch
import torch.nn as nn

from gatetracker.losses import (
    EpipolarLoss,
    WeightedSmoothL1Loss,
    WeightedCombinationLoss,
    GateTrackerLoss,
    InfoNCELoss,
)


def test_EpipolarLoss_correct_F():
    epipolar_loss_fn = EpipolarLoss()

    batch_size = 4
    num_points = 50

    F_pred = torch.randn(batch_size, 3, 3)
    F_pred = F_pred / F_pred.norm(dim=(1, 2), keepdim=True)

    pts1 = torch.randn(batch_size, num_points, 2)
    pts2 = torch.randn(batch_size, num_points, 2)

    scores = torch.ones(batch_size, num_points)

    loss = epipolar_loss_fn(F_pred, pts1, pts2, scores)
    assert loss.shape == ()
    assert loss.item() >= 0


def test_EpipolarLoss_backward():
    epipolar_loss_fn = EpipolarLoss()

    F_pred = torch.randn(2, 3, 3, requires_grad=True)
    pts1 = torch.randn(2, 10, 2)
    pts2 = torch.randn(2, 10, 2)
    scores = torch.ones(2, 10)

    loss = epipolar_loss_fn(F_pred, pts1, pts2, scores)
    loss.backward()
    assert F_pred.grad is not None
    assert torch.isfinite(F_pred.grad).all()


def test_WeightedSmoothL1Loss():
    loss_fn = WeightedSmoothL1Loss(beta=1.0)
    pred = torch.randn(10, 2)
    target = torch.randn(10, 2)
    weights = torch.ones(10)
    valid_mask = torch.ones(10, dtype=torch.bool)

    loss = loss_fn(pred, target, weights, valid_mask)
    assert loss.shape == ()
    assert loss.item() >= 0


def test_InfoNCELoss_basic():
    loss_fn = InfoNCELoss(temperature=0.1, symmetric=False)
    logits = torch.randn(8, 8)
    positives = torch.arange(8)
    mask = torch.ones(8, 8, dtype=torch.bool)

    loss = loss_fn(
        source_to_target_logits=logits,
        source_to_target_positive=positives,
        source_to_target_mask=mask,
    )
    assert loss.shape == ()
    assert loss.item() > 0


def test_GateTrackerLoss_instantiation():
    from dotmap import DotMap
    config = DotMap({
        "INFONCE_TEMPERATURE": 0.07,
        "INFONCE_SYMMETRIC": True,
        "FINE_REFINEMENT_LOSS_WEIGHT": 0.25,
        "FINE_REFINEMENT_LOSS_BETA": 1.0,
        "LOSS_WEIGHT": {"Fundamental": 0.8},
        "TRACKING_POSITION_LOSS_WEIGHT": 1.0,
        "TRACKING_VISIBILITY_LOSS_WEIGHT": 0.5,
    })
    loss_obj = GateTrackerLoss(config)
    assert loss_obj is not None
