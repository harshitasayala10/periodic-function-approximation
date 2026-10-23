# Periodic Function Approximation with SIREN

Take-home prototype: learn a **bounded, periodic** black-box function \(f(x)\) from data, training only on \(x \in [-10\pi, 10\pi]\), and evaluate **extrapolation** outside that range.

## Approach (why SIREN)

Standard MLPs with ReLU or Tanh activations do not keep oscillating outside the training interval: ReLU tends toward linear rays; Tanh saturates. A periodic target needs a representation that can continue to oscillate for large \(|x|\).

**SIREN** (Sitzmann et al., NeurIPS 2020) uses \(\sin(\omega_0 \cdot (Wx + b))\) activations with a carefully designed initialization. Sinusoids oscillate everywhere by construction, so the network can represent periodic structure without hardcoding the unknown period \(T\) or frequency into features or the loss. Frequencies are learned from \((x, f(x))\) pairs only.

We reject alternatives that either need a known period (periodic GP kernels, hand-built Fourier bases with fixed \(T\)) or fail at extrapolation (vanilla ReLU/Tanh MLPs).

**Constraints honored**

- Train only on \(x \in [-10\pi, 10\pi]\).
- The model never calls \(f\) at inference — only `model(x)`.
- No hardcoded period/frequency of the target in the architecture, features, or loss.
- Ground-truth \(f\) is used for **data generation** and for **scoring** eval grids / plots only (`make_eval_grid`).

Placeholder target (swap in `src/black_box.py` when the real \(f\) is provided):

\[
f(x) = \sin(x) + 0.5\sin(2x)
\]

## Project layout

```
src/
  black_box.py   # ground-truth f (data / metrics only)
  data.py        # sampling + datasets
  model.py       # SIREN
  train.py       # training + checkpointing
  evaluate.py    # MSE + plots
run.py           # CLI entry point
tests/           # pytest suite
outputs/         # checkpoints + plots (gitignored artifacts)
```

## Setup

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce training and evaluation

**Full training** (default plan: 500 epochs, width 256 × 4 layers ≈ 198k params):

```bash
python run.py
```

`python run.py` writes a local checkpoint at `outputs/checkpoints/best_model.pt`. That weight file is not part of the repository. The figure from the submitted run is committed at `outputs/plots/siren_evaluation.png`.

**Fast smoke run** (recommended for a quick check on CPU):

```bash
python run.py --epochs 20 --hidden-dims 32 32 --n-train 512 --n-val 128 --batch-size 64
```

### Evaluation results

Full run: `python run.py`, seed 42, 5,000 stratified training points, 1,000 validation points, four hidden layers of 256 (198,145 parameters), 500 epochs, Adam at \(10^{-4}\). Best validation MSE was \(7.2 \times 10^{-8}\). The committed figure is `outputs/plots/siren_evaluation.png`. Re-running `python run.py` overwrites the local checkpoint and can regenerate that figure.

| Region | MSE | MAE | R² |
|--------|-----|-----|----|
| In-distribution \([-10\pi, 10\pi]\) | \(7.0 \times 10^{-8}\) | 0.000187 | 1.000 |
| Near, full \([-20\pi, 20\pi]\) | 0.102 | 0.180 | 0.837 |
| Far, full \([-50\pi, 50\pi]\) | 0.232 | 0.349 | 0.628 |
| Near, outside training only | 0.204 | 0.360 | 0.674 |
| Far, outside training only | 0.319 | 0.462 | 0.489 |

Inside the training window the fit is essentially exact. Error grows once \(x\) leaves that window. The full near and far intervals look better than the outside-only rows because they still include the training window.

## Tests

```bash
pytest -q
```

Coverage includes sampling domain/strategies, dataset shapes, SIREN forward shapes / init / param count, short training + checkpointing, and evaluation returning finite MSE keys.

## Design choices (brief)

| Choice | Rationale |
|--------|-----------|
| Stratified sampling, 5k points | Coverage across ~10 periods in the training window without large gaps |
| MSE + Adam \(10^{-4}\) | Standard for clean regression / SIREN |
| `ReduceLROnPlateau` | Lower LR when validation stalls |
| Checkpoint best val loss | Prefer generalization over last epoch |
| \(\omega_0 = 30\) | Default from the SIREN paper |

## Limitations / failure cases

- **Period mismatch / multi-scale**: if the true \(f\) has very high-frequency content relative to \(\omega_0\) and width, fitting may need more capacity or longer training.
- **Extrapolation drifts**: on this run, outside-only R² falls from 0.67 just beyond the training window to 0.49 on the far bands. Sine layers keep oscillating, but they are not locked to the unknown period.
- **Placeholder \(f\)**: results on \(\sin x + 0.5\sin 2x\) are a proxy; the real black box may differ.
- **No period discovery report**: we do not estimate \(T\) explicitly (by design - not hardcoded, not returned).

## What I would improve with more time

- Ablations on width, depth, \(\omega_0\), and sample count.
- A learned input warp or post-hoc period estimate, checked only at analysis time, to stop the far-region drift.
- Optional period estimation *post hoc* from predictions (analysis only, not baked into the model).
- Hyperparameter search and early stopping with a held-out extrapolation split.
- Packaging (`pyproject.toml`) and a small inference-only CLI that loads a checkpoint and never imports `black_box`.
