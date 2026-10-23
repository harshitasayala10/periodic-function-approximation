"""Evaluation returns finite errors and writes a plot without a display."""

import math

import pytest
import torch

from src.evaluate import evaluate
from src.model import SIREN

METRIC_NAMES = (
    "in_distribution",
    "near",
    "far",
    "near_outside",
    "far_outside",
)


def test_evaluate_returns_finite_metrics(tmp_path):
    torch.manual_seed(0)
    model = SIREN(hidden_dims=[8, 8], omega_0=30.0)
    results = evaluate(
        model,
        plot_dir=str(tmp_path),
        device=torch.device("cpu"),
        n_points=64,
    )

    for name in METRIC_NAMES:
        for prefix in ("mse", "mae", "r2"):
            key = f"{prefix}_{name}"
            value = results[key]
            assert isinstance(value, float)
            assert math.isfinite(value)

    plot = tmp_path / "siren_evaluation.png"
    assert plot.is_file()
    assert results["plot_path"] == str(plot)
    assert results["near_in_units_of_pi"] == pytest.approx(20.0)
    assert results["far_in_units_of_pi"] == pytest.approx(50.0)
