"""Ablation study for the periodic-function SIREN.

Each group changes one choice and retrains under the same budget as
``train``: 5,000 points, Adam at 1e-4, the plateau schedule, and the
weights with the best validation loss. The baseline SIREN is trained
once and reused in every group.

Zones match ``evaluate``:

- in-distribution: [-10π, 10π]
- near, outside training: [-20π, -10π] and [10π, 20π]
- far, outside training: [-50π, -20π] and [20π, 50π]

ReLU and Tanh use PyTorch's default initialization. SIREN uses the
paper initialization. The activation comparison is therefore activation
plus the initialization that belongs to it.
"""

import csv
import math
import os
from typing import Callable, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .data import TRAIN_X_MAX, TRAIN_X_MIN
from .evaluate import FAR_PI, NEAR_PI, _concat, _predict_grid, _r2
from .model import SIREN, ReLUNet, TanhNet
from .train import set_seed, train_model

BASE_DIMS = [256, 256, 256, 256]
CSV_FIELDS = [
    "group",
    "label",
    "best_val_mse",
    "in_dist_r2",
    "near_outside_r2",
    "far_outside_r2",
]


def score_model(
    model: torch.nn.Module,
    device: torch.device,
    n_points: int = 2000,
) -> Dict[str, float]:
    """R² on the training window and the two outside-only bands."""
    model.eval()
    model.to(device)
    near = NEAR_PI * math.pi
    far = FAR_PI * math.pi

    _, y_in, pred_in = _predict_grid(
        model, TRAIN_X_MIN, TRAIN_X_MAX, n_points, device
    )
    near_parts = [
        _predict_grid(model, -near, TRAIN_X_MIN, n_points, device),
        _predict_grid(model, TRAIN_X_MAX, near, n_points, device),
    ]
    _, y_near, pred_near = _concat(near_parts)
    far_parts = [
        _predict_grid(model, -far, -near, n_points, device),
        _predict_grid(model, near, far, n_points, device),
    ]
    _, y_far, pred_far = _concat(far_parts)
    return {
        "in_dist_r2": _r2(pred_in, y_in),
        "near_outside_r2": _r2(pred_near, y_near),
        "far_outside_r2": _r2(pred_far, y_far),
    }


def _append_row(path: str, row: Dict[str, object], write_header: bool) -> None:
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row[key] for key in CSV_FIELDS})


def _format_row(group: str, label: str, scores: Dict[str, float]) -> Dict[str, object]:
    return {
        "group": group,
        "label": label,
        "best_val_mse": f"{scores['best_val_mse']:.6e}",
        "in_dist_r2": f"{scores['in_dist_r2']:.4f}",
        "near_outside_r2": f"{scores['near_outside_r2']:.4f}",
        "far_outside_r2": f"{scores['far_outside_r2']:.4f}",
    }


def _train_and_score(
    builder: Callable[[], torch.nn.Module],
    device: torch.device,
    n_epochs: int,
    strategy: str,
    seed: int,
    n_points: int,
) -> Dict[str, float]:
    set_seed(seed)
    model = builder()
    best_val = train_model(
        model,
        n_epochs=n_epochs,
        strategy=strategy,
        seed=seed,
        device=device,
        verbose=False,
    )
    scores = score_model(model, device, n_points=n_points)
    scores["best_val_mse"] = best_val
    return scores


def save_plot(rows: List[Dict[str, object]], path: str) -> str:
    """Bar chart of R². Each panel scales to its own data, including negatives."""
    groups: List[str] = []
    for row in rows:
        if row["group"] not in groups:
            groups.append(str(row["group"]))

    zone_keys = ["in_dist_r2", "near_outside_r2", "far_outside_r2"]
    zone_labels = ["In-distribution", "Near outside", "Far outside"]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, axes = plt.subplots(1, len(groups), figsize=(4.6 * len(groups), 5.2))
    if len(groups) == 1:
        axes = [axes]

    for ax, group in zip(axes, groups):
        group_rows = [row for row in rows if row["group"] == group]
        labels = [str(row["label"]) for row in group_rows]
        bar_width = 0.25
        positions = list(range(len(labels)))
        parsed = {
            key: [float(row[key]) for row in group_rows] for key in zone_keys
        }
        for index, (key, zone_label, color) in enumerate(
            zip(zone_keys, zone_labels, colors)
        ):
            offsets = [pos + index * bar_width for pos in positions]
            ax.bar(
                offsets,
                parsed[key],
                width=bar_width,
                label=zone_label,
                color=color,
                alpha=0.9,
            )
        flat = [value for key in zone_keys for value in parsed[key]]
        low = min(flat)
        high = max(flat)
        pad = 0.08 * (high - low) if high > low else 0.05
        ax.set_ylim(low - pad, high + pad)
        ax.axhline(0.0, color="black", linewidth=0.6, linestyle="--")
        ax.set_title(group)
        ax.set_xticks([pos + bar_width for pos in positions])
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("R²" if ax is axes[0] else "")
        ax.legend(fontsize=7)

    plt.suptitle("Ablation: R² on the submitted evaluation zones")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def run_ablation(
    out_dir: str = "outputs/ablation",
    n_epochs: int = 500,
    seed: int = 42,
    n_points: int = 2000,
) -> List[Dict[str, object]]:
    """Run every variant, appending each result to the CSV as it finishes."""
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "results.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    cache: Dict[str, Dict[str, float]] = {}
    written: List[Dict[str, object]] = []

    def run_key(
        key: str,
        builder: Callable[[], torch.nn.Module],
        strategy: str = "stratified",
    ) -> Dict[str, float]:
        if key not in cache:
            print(f"  Training: {key}")
            cache[key] = _train_and_score(
                builder, device, n_epochs, strategy, seed, n_points
            )
            scores = cache[key]
            print(
                f"    val_mse={scores['best_val_mse']:.3e} | "
                f"in={scores['in_dist_r2']:.3f} | "
                f"near={scores['near_outside_r2']:.3f} | "
                f"far={scores['far_outside_r2']:.3f}"
            )
        return cache[key]

    specs = [
        ("Activation", "SIREN (sin, baseline)", "siren-base", lambda: SIREN(hidden_dims=BASE_DIMS, omega_0=30.0), "stratified"),
        ("Activation", "ReLU", "relu", lambda: ReLUNet(hidden_dims=BASE_DIMS), "stratified"),
        ("Activation", "Tanh", "tanh", lambda: TanhNet(hidden_dims=BASE_DIMS), "stratified"),
        ("ω₀", "ω₀ = 1", "omega-1", lambda: SIREN(hidden_dims=BASE_DIMS, omega_0=1.0), "stratified"),
        ("ω₀", "ω₀ = 10", "omega-10", lambda: SIREN(hidden_dims=BASE_DIMS, omega_0=10.0), "stratified"),
        ("ω₀", "ω₀ = 30 (baseline)", "siren-base", lambda: SIREN(hidden_dims=BASE_DIMS, omega_0=30.0), "stratified"),
        ("ω₀", "ω₀ = 100", "omega-100", lambda: SIREN(hidden_dims=BASE_DIMS, omega_0=100.0), "stratified"),
        ("Depth", "1 layer", "depth-1", lambda: SIREN(hidden_dims=[256], omega_0=30.0), "stratified"),
        ("Depth", "2 layers", "depth-2", lambda: SIREN(hidden_dims=[256, 256], omega_0=30.0), "stratified"),
        ("Depth", "4 layers (baseline)", "siren-base", lambda: SIREN(hidden_dims=BASE_DIMS, omega_0=30.0), "stratified"),
        ("Depth", "6 layers", "depth-6", lambda: SIREN(hidden_dims=[256] * 6, omega_0=30.0), "stratified"),
        ("Sampling", "Stratified (baseline)", "siren-base", lambda: SIREN(hidden_dims=BASE_DIMS, omega_0=30.0), "stratified"),
        ("Sampling", "Uniform", "sample-uniform", lambda: SIREN(hidden_dims=BASE_DIMS, omega_0=30.0), "uniform"),
        ("Sampling", "Grid", "sample-grid", lambda: SIREN(hidden_dims=BASE_DIMS, omega_0=30.0), "grid"),
    ]

    for index, (group, label, key, builder, strategy) in enumerate(specs):
        print(f"\n── {group}: {label} ──")
        scores = run_key(key, builder, strategy)
        row = _format_row(group, label, scores)
        _append_row(csv_path, row, write_header=index == 0)
        written.append(row)
        print(f"  recorded → {csv_path}")

    plot_path = os.path.join(out_dir, "ablation_plot.png")
    save_plot(written, plot_path)
    print(f"\nCSV saved  → {csv_path}")
    print(f"Plot saved → {plot_path}")
    return written
