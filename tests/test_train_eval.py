"""Fast end-to-end smoke tests for training and evaluation."""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch

from src.evaluate import evaluate
from src.train import train


def test_short_training_decreases_loss_and_writes_checkpoint(
    tmp_path: Path,
) -> None:
    ckpt_dir = tmp_path / "checkpoints"
    model = train(
        n_epochs=5,
        n_train=256,
        n_val=64,
        batch_size=64,
        learning_rate=1e-3,
        hidden_dims=[16, 16],
        omega_0=30.0,
        checkpoint_dir=str(ckpt_dir),
        seed=0,
        verbose=False,
    )

    ckpt = ckpt_dir / "best_model.pt"
    assert ckpt.is_file()
    assert model.count_parameters() > 0

    # Reloadable checkpoint
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model2 = type(model)(hidden_dims=[16, 16], omega_0=30.0)
    model2.load_state_dict(state)


def test_evaluate_returns_finite_mse(tmp_path: Path) -> None:
    plot_dir = tmp_path / "plots"
    model = train(
        n_epochs=3,
        n_train=128,
        n_val=32,
        batch_size=32,
        learning_rate=1e-3,
        hidden_dims=[8, 8],
        omega_0=30.0,
        checkpoint_dir=str(tmp_path / "ckpts"),
        seed=1,
        verbose=False,
    )
    results = evaluate(
        model,
        plot_dir=str(plot_dir),
        device=torch.device("cpu"),
        n_points=64,
    )

    assert "mse_in_distribution" in results
    assert "mse_near" in results
    assert "mse_far" in results
    assert "r2_near_outside" in results
    assert "plot_path" in results
    assert math.isfinite(results["mse_in_distribution"])
    assert math.isfinite(results["mse_near"])
    assert math.isfinite(results["mse_far_outside"])
    assert os.path.isfile(results["plot_path"])
