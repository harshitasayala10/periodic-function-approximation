"""Training loop for a SIREN on samples of a periodic function."""

import copy
import json
import os
import random
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .data import make_dataloaders
from .model import SIREN


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch. Dataset sampling uses its own Generator."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_state_dict(path: str, device: torch.device) -> dict:
    """Load a checkpoint onto ``device``.

    ``weights_only=True`` avoids unpickling arbitrary objects. Older PyTorch
    builds that do not accept the argument fall back to ``map_location`` only.
    """
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def train(
    n_epochs: int = 500,
    n_train: int = 5000,
    n_val: int = 1000,
    batch_size: int = 256,
    learning_rate: float = 1e-4,
    hidden_dims: Optional[Sequence[int]] = None,
    omega_0: float = 30.0,
    checkpoint_dir: str = "outputs/checkpoints",
    seed: int = 42,
    verbose: bool = True,
) -> SIREN:
    """Train a SIREN and return the weights with the best validation loss.

    The returned module has a ``history`` attribute: a list of per-epoch
    ``train_loss`` and ``val_loss`` values. It also stores ``best_val_loss``.
    """
    if hidden_dims is None:
        hidden_dims = [256, 256, 256, 256]
    dims: List[int] = list(hidden_dims)

    set_seed(seed)
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Training on: {device}")

    train_loader, val_loader = make_dataloaders(
        n_train=n_train,
        n_val=n_val,
        batch_size=batch_size,
        seed=seed,
    )

    model = SIREN(hidden_dims=dims, omega_0=omega_0).to(device)
    if verbose:
        print(f"Parameters: {model.count_parameters():,}")

    loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    # ``verbose`` was removed from ReduceLROnPlateau in recent PyTorch.
    scheduler = ReduceLROnPlateau(optimizer, patience=50, factor=0.5)

    best_val_loss = float("inf")
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
    history = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(x_batch)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = model(x_batch)
                val_loss += loss_fn(y_pred, y_batch).item() * len(x_batch)

        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)

        should_print = epoch == 1 or epoch == n_epochs or epoch % 50 == 0
        if verbose and should_print:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:4d}/{n_epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Best: {best_val_loss:.6f} | "
                f"LR: {current_lr:.2e}"
            )

    if os.path.exists(checkpoint_path):
        model.load_state_dict(_load_state_dict(checkpoint_path, device))

    config = {
        "n_epochs": n_epochs,
        "n_train": n_train,
        "n_val": n_val,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "hidden_dims": dims,
        "omega_0": omega_0,
        "seed": seed,
        "best_val_loss": best_val_loss,
        "history": history,
    }
    config_path = os.path.join(checkpoint_dir, "train_config.json")
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    model.history = history
    model.best_val_loss = best_val_loss
    if verbose:
        print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
        print(f"Checkpoint: {checkpoint_path}")
    return model


def train_model(
    model: nn.Module,
    n_epochs: int = 500,
    n_train: int = 5000,
    n_val: int = 1000,
    batch_size: int = 256,
    learning_rate: float = 1e-4,
    strategy: str = "stratified",
    seed: int = 42,
    device: torch.device = None,
    verbose: bool = False,
) -> float:
    """Train ``model`` in place and restore the best validation weights.

    Same loss, optimizer, and learning-rate schedule as ``train``. The
    returned model is the checkpoint with the lowest validation MSE, not
    the weights from the final epoch. ``strategy`` selects the sampler in
    ``make_dataloaders``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(seed)
    train_loader, val_loader = make_dataloaders(
        n_train=n_train,
        n_val=n_val,
        strategy=strategy,
        batch_size=batch_size,
        seed=seed,
    )

    model = model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, patience=50, factor=0.5)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, n_epochs + 1):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                val_loss += loss_fn(model(x_batch), y_batch).item() * len(x_batch)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        if verbose and (epoch == 1 or epoch == n_epochs or epoch % 100 == 0):
            print(f"  epoch {epoch:4d} | val_loss={val_loss:.6f}")

    model.load_state_dict(best_state)
    model.best_val_loss = best_val_loss
    return best_val_loss
