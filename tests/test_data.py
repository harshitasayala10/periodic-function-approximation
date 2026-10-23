"""Sampling, domain bounds, and dataset shapes."""

import math

import numpy as np
import pytest
import torch

from src.black_box import f
from src.data import (
    TRAIN_X_MAX,
    TRAIN_X_MIN,
    PeriodicFunctionDataset,
    make_dataloaders,
)


def test_training_domain_is_ten_pi():
    assert TRAIN_X_MIN == pytest.approx(-10.0 * math.pi)
    assert TRAIN_X_MAX == pytest.approx(10.0 * math.pi)


@pytest.mark.parametrize("strategy", ["uniform", "grid", "stratified"])
def test_samples_stay_inside_training_domain(strategy):
    ds = PeriodicFunctionDataset(n_samples=400, strategy=strategy, seed=7)
    xs = ds.x.numpy().reshape(-1)
    assert xs.min() >= TRAIN_X_MIN - 1e-6
    assert xs.max() <= TRAIN_X_MAX + 1e-6


def test_grid_includes_endpoints_and_is_sorted():
    n = 25
    ds = PeriodicFunctionDataset(n_samples=n, strategy="grid", seed=0)
    xs = ds.x.numpy().reshape(-1)
    assert xs.shape == (n,)
    assert xs[0] == pytest.approx(TRAIN_X_MIN)
    assert xs[-1] == pytest.approx(TRAIN_X_MAX)
    assert np.all(np.diff(xs) > 0)


def test_stratified_places_one_point_in_each_bin():
    n = 40
    ds = PeriodicFunctionDataset(n_samples=n, strategy="stratified", seed=3)
    xs = ds.x.numpy().reshape(-1)
    bins = np.linspace(TRAIN_X_MIN, TRAIN_X_MAX, n + 1)
    assert np.all(xs >= bins[:-1] - 1e-8)
    assert np.all(xs <= bins[1:] + 1e-8)
    # Points are generated in bin order, so each slice contains its own sample.
    assert np.all(xs < bins[1:] + 1e-8)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="Unknown strategy"):
        PeriodicFunctionDataset(n_samples=10, strategy="random-walk")


def test_dataset_shapes_and_targets():
    ds = PeriodicFunctionDataset(n_samples=12, strategy="grid", seed=1)
    assert len(ds) == 12
    assert ds.x.shape == (12, 1)
    assert ds.y.shape == (12, 1)
    assert ds.x.dtype == torch.float32
    assert ds.y.dtype == torch.float32
    x0, y0 = ds[0]
    assert x0.shape == (1,)
    assert y0.shape == (1,)
    for x, y in zip(ds.x.reshape(-1), ds.y.reshape(-1)):
        assert y.item() == pytest.approx(f(float(x.item())), abs=1e-5)


def test_same_seed_is_reproducible_and_strategies_differ():
    a = PeriodicFunctionDataset(n_samples=30, strategy="uniform", seed=5)
    b = PeriodicFunctionDataset(n_samples=30, strategy="uniform", seed=5)
    c = PeriodicFunctionDataset(n_samples=30, strategy="uniform", seed=6)
    assert torch.equal(a.x, b.x)
    assert torch.equal(a.y, b.y)
    assert not torch.equal(a.x, c.x)


def test_dataloaders_batch_and_split_seeds():
    train_loader, val_loader = make_dataloaders(
        n_train=100, n_val=40, batch_size=16, seed=9
    )
    xb, yb = next(iter(train_loader))
    assert xb.shape == (16, 1)
    assert yb.shape == (16, 1)
    assert len(train_loader.dataset) == 100
    assert len(val_loader.dataset) == 40
    assert not torch.equal(train_loader.dataset.x[:40], val_loader.dataset.x)
