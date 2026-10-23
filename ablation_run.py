"""Run the ablation study.

Usage:
    python ablation_run.py
    python ablation_run.py --epochs 500
"""

import argparse

from src.ablation import run_ablation


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation study for the SIREN prototype.")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="outputs/ablation")
    args = parser.parse_args()

    print("=" * 60)
    print("Ablation study")
    print("=" * 60)
    run_ablation(out_dir=args.out_dir, n_epochs=args.epochs, seed=args.seed)
    print(f"\nDone. Results are in {args.out_dir}/")


if __name__ == "__main__":
    main()
