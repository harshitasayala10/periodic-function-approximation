"""Ablation helpers return finite scores and matching parameter counts."""

import math

import torch

from src.ablation import score_model
from src.model import SIREN, ReLUNet, TanhNet
from src.train import train_model


def test_matched_architectures_have_the_same_parameter_count():
    dims = [32, 32]
    siren = SIREN(hidden_dims=dims, omega_0=30.0)
    relu = ReLUNet(hidden_dims=dims)
    tanh = TanhNet(hidden_dims=dims)
    assert siren.count_parameters() == relu.count_parameters()
    assert relu.count_parameters() == tanh.count_parameters()


def test_score_model_returns_finite_zone_scores():
    torch.manual_seed(0)
    model = SIREN(hidden_dims=[8, 8], omega_0=30.0)
    scores = score_model(model, torch.device("cpu"), n_points=32)
    for key in ("in_dist_r2", "near_outside_r2", "far_outside_r2"):
        assert key in scores
        assert isinstance(scores[key], float)
        assert math.isfinite(scores[key])


def test_train_model_restores_best_validation_loss():
    torch.manual_seed(0)
    model = ReLUNet(hidden_dims=[8, 8])
    best = train_model(
        model,
        n_epochs=3,
        n_train=64,
        n_val=32,
        batch_size=32,
        learning_rate=1e-3,
        seed=0,
        device=torch.device("cpu"),
        verbose=False,
    )
    assert math.isfinite(best)
    assert model.best_val_loss == best
    scores = score_model(model, torch.device("cpu"), n_points=32)
    assert math.isfinite(scores["in_dist_r2"])
