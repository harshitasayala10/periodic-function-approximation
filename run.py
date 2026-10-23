"""Train a SIREN on the black-box function and evaluate it.

Full training (plan defaults):

    python run.py

Short smoke run:

    python run.py --epochs 20 --hidden-dims 32 32 --n-train 512 --n-val 128 --batch-size 64
"""

import argparse

import torch

from src.evaluate import evaluate
from src.train import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a SIREN on the black-box periodic function."
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--n-val", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[256, 256, 256, 256],
    )
    parser.add_argument("--omega-0", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    parser.add_argument("--plot-dir", default="outputs/plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 60)
    print("Andromeda take-home: SIREN training")
    print("=" * 60)

    model = train(
        n_epochs=args.epochs,
        n_train=args.n_train,
        n_val=args.n_val,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_dims=args.hidden_dims,
        omega_0=args.omega_0,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("Evaluation")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = evaluate(model, plot_dir=args.plot_dir, device=device)

    print("\nFinal results:")
    for name in (
        "in_distribution",
        "near",
        "far",
        "near_outside",
        "far_outside",
    ):
        print(
            f"  {name:16s}  "
            f"MSE {results[f'mse_{name}']:.6f}  "
            f"MAE {results[f'mae_{name}']:.6f}  "
            f"R2 {results[f'r2_{name}']:.4f}"
        )
    print(f"  Plot: {results['plot_path']}")


if __name__ == "__main__":
    main()
