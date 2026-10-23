"""Metrics and plots for a trained SIREN.

Ground truth is queried only to score and plot the model, via
``make_eval_grid``. Predictions themselves go through ``predict``, which
does not call ``f``.
"""

import math
import os
from typing import List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .data import TRAIN_X_MAX, TRAIN_X_MIN, make_eval_grid
from .model import SIREN, predict

# Named evaluation windows, in units of pi. The training window is 10.
NEAR_PI = 20.0
FAR_PI = 50.0


def _mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean((pred - target) ** 2).item()


def _mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean(torch.abs(pred - target)).item()


def _r2(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Coefficient of determination on this grid.

    A model that predicts the mean of the targets scores 0. The score is
    undefined when the targets do not vary, which does not happen for a
    non-constant periodic function on these windows.
    """
    ss_res = torch.sum((pred - target) ** 2)
    ss_tot = torch.sum((target - torch.mean(target)) ** 2)
    if ss_tot.item() == 0.0:
        return 1.0 if ss_res.item() == 0.0 else 0.0
    return (1.0 - ss_res / ss_tot).item()


def _points_for_width(x_min: float, x_max: float, n_points: int) -> int:
    """Keep the same point density as ``n_points`` across the training window."""
    train_width = TRAIN_X_MAX - TRAIN_X_MIN
    width = x_max - x_min
    return max(2, int(round(n_points * width / train_width)))


def _score(pred: torch.Tensor, target: torch.Tensor) -> dict:
    return {
        "mse": _mse(pred, target),
        "mae": _mae(pred, target),
        "r2": _r2(pred, target),
    }


def _predict_grid(
    model: SIREN,
    x_min: float,
    x_max: float,
    n_points: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count = _points_for_width(x_min, x_max, n_points)
    x, y_true = make_eval_grid(x_min, x_max, n_points=count)
    y_pred = predict(model, x.to(device)).cpu()
    return x, y_true, y_pred


def _concat(
    parts: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs, trues, preds = zip(*parts)
    return torch.cat(list(xs)), torch.cat(list(trues)), torch.cat(list(preds))


def _record(results: dict, name: str, pred: torch.Tensor, target: torch.Tensor) -> None:
    scores = _score(pred, target)
    results[f"mse_{name}"] = scores["mse"]
    results[f"mae_{name}"] = scores["mae"]
    results[f"r2_{name}"] = scores["r2"]
    print(
        f"{name:16s}  MSE {scores['mse']:.6f}  "
        f"MAE {scores['mae']:.6f}  R2 {scores['r2']:.4f}"
    )


def _plot_zone(ax, x, y_true, y_pred, title: str) -> None:
    x_np = x.numpy().flatten()
    y_true_np = y_true.numpy().flatten()
    y_pred_np = y_pred.numpy().flatten()
    span = float(y_true_np.max() - y_true_np.min())
    pad = 0.1 * span if span > 0 else 0.1
    y_min = float(y_true_np.min() - pad)
    y_max = float(y_true_np.max() + pad)

    ax.plot(x_np, y_true_np, label="True f(x)", color="blue", alpha=0.7)
    ax.plot(
        x_np,
        y_pred_np,
        label="SIREN prediction",
        color="red",
        linestyle="--",
        alpha=0.7,
    )
    if x_np.min() <= TRAIN_X_MIN <= x_np.max():
        ax.axvline(x=TRAIN_X_MIN, color="gray", linestyle=":", label="Training boundary")
    if x_np.min() <= TRAIN_X_MAX <= x_np.max():
        ax.axvline(x=TRAIN_X_MAX, color="gray", linestyle=":")
    if x_np.min() < TRAIN_X_MAX and x_np.max() > TRAIN_X_MIN:
        ax.fill_betweenx(
            [y_min, y_max],
            max(float(x_np.min()), TRAIN_X_MIN),
            min(float(x_np.max()), TRAIN_X_MAX),
            alpha=0.05,
            color="green",
            label="Training region",
        )
    ax.set_ylim(y_min, y_max)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend()


def evaluate(
    model: SIREN,
    plot_dir: str = "outputs/plots",
    device: torch.device = None,
    n_points: int = 2000,
) -> dict:
    """Score the model on the training window and two wider windows.

    Full intervals, which include the training window:

    - in-distribution: [-10π, 10π]
    - near: [-20π, 20π]
    - far: [-50π, 50π]

    Outside-only intervals, which exclude points the model was trained on:

    - near outside: [-20π, -10π] and [10π, 20π]
    - far outside: [-50π, -20π] and [20π, 50π]

    Each region gets MSE, MAE, and R². The figure has one panel per full interval.
    """
    if device is None:
        device = next(model.parameters()).device

    os.makedirs(plot_dir, exist_ok=True)
    model.eval()
    model.to(device)

    near_max = NEAR_PI * math.pi
    far_max = FAR_PI * math.pi
    near_min = -near_max
    far_min = -far_max

    zones: List[Tuple[str, float, float]] = [
        ("in_distribution", TRAIN_X_MIN, TRAIN_X_MAX),
        ("near", near_min, near_max),
        ("far", far_min, far_max),
    ]
    grids = {}
    results = {
        "train_x_min": TRAIN_X_MIN,
        "train_x_max": TRAIN_X_MAX,
        "near_in_units_of_pi": NEAR_PI,
        "far_in_units_of_pi": FAR_PI,
    }

    for name, x_min, x_max in zones:
        grids[name] = _predict_grid(model, x_min, x_max, n_points, device)
        _record(results, name, grids[name][2], grids[name][1])

    near_left = _predict_grid(model, near_min, TRAIN_X_MIN, n_points, device)
    near_right = _predict_grid(model, TRAIN_X_MAX, near_max, n_points, device)
    _, near_out_true, near_out_pred = _concat([near_left, near_right])
    _record(results, "near_outside", near_out_pred, near_out_true)

    far_left = _predict_grid(model, far_min, near_min, n_points, device)
    far_right = _predict_grid(model, near_max, far_max, n_points, device)
    _, far_out_true, far_out_pred = _concat([far_left, far_right])
    _record(results, "far_outside", far_out_pred, far_out_true)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    titles = {
        "in_distribution": "In-distribution [-10π, 10π]",
        "near": "Near window [-20π, 20π]",
        "far": "Far window [-50π, 50π]",
    }
    for ax, name in zip(axes, ("in_distribution", "near", "far")):
        x, y_true, y_pred = grids[name]
        _plot_zone(ax, x, y_true, y_pred, titles[name])

    plt.tight_layout()
    plot_path = os.path.join(plot_dir, "siren_evaluation.png")
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to: {plot_path}")

    results["plot_path"] = plot_path
    return results
