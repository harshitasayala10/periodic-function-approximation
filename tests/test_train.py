"""A few epochs of a tiny SIREN should learn and write a checkpoint."""

import os

from src.train import train


def test_short_training_decreases_loss_and_writes_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    model = train(
        n_epochs=12,
        n_train=256,
        n_val=64,
        batch_size=64,
        learning_rate=1e-3,
        hidden_dims=[16, 16],
        omega_0=30.0,
        checkpoint_dir=str(checkpoint_dir),
        seed=0,
        verbose=False,
    )

    checkpoint = checkpoint_dir / "best_model.pt"
    assert checkpoint.is_file()
    assert os.path.getsize(checkpoint) > 0
    assert len(model.history) == 12

    train_losses = [row["train_loss"] for row in model.history]
    val_losses = [row["val_loss"] for row in model.history]
    assert all(loss == loss for loss in train_losses)  # not NaN
    assert all(loss == loss for loss in val_losses)
    assert min(train_losses) < train_losses[0]
    assert model.best_val_loss <= min(val_losses) + 1e-12
    assert model.best_val_loss < val_losses[0]
