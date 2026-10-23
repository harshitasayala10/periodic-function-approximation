# Periodic Function Approximation with SIREN

Learn a **bounded, periodic** function \(f(x)\) from examples on \(x \in [-10\pi, 10\pi]\), then test the trained model beyond that window.

A SIREN fit to those examples matches the training window essentially exactly. Sine activations let the prediction keep oscillating for inputs the model was not trained on. Metrics, plots, and an ablation study are below.

## Approach (why SIREN)

ReLU becomes a straight line outside the training inputs. Tanh saturates and stops oscillating. A periodic target needs a representation that can keep moving for large \(|x|\).

**SIREN** (Sitzmann et al., NeurIPS 2020) uses \(\sin(\omega_0 \cdot (Wx + b))\) with the paper's initialization. Frequencies are learned from \((x, f(x))\) pairs. The period \(T\) is not hardcoded. Among the models compared in the ablation, this was the best joint result: it fits the training window and has the highest R² just outside it. It is not a solution for every real \(x\).

A periodic Gaussian process or a fixed Fourier basis would need \(T\) set in advance, which the task does not allow. Those options were not used.

**Constraints honored**

- Train only on \(x \in [-10\pi, 10\pi]\).
- The model never calls \(f\) at inference — only `model(x)`.
- No hardcoded period/frequency of the target in the architecture, features, or loss.
- Ground-truth \(f\) is used for **data generation** and for **scoring** eval grids / plots only (`make_eval_grid`).

The target chosen for this submission is

\[
f(x) = \sin(x) + 0.5\sin(2x)
\]

## Project layout

```
src/
  black_box.py   # ground-truth f (data / metrics only)
  data.py        # sampling + datasets
  model.py       # SIREN, plus ReLU and Tanh networks used in the ablation
  train.py       # training + checkpointing
  evaluate.py    # MSE, MAE, R², and plots
  ablation.py    # one-factor ablation study
run.py           # train and evaluate the submitted model
ablation_run.py # reproduce the ablation
tests/           # pytest suite
outputs/         # checkpoint (local), evaluation plot, ablation results
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

## Ablation

One factor was changed at a time. Every variant used the same budget as the submitted run: 5,000 points, 1,000 validation points, 500 epochs, Adam at \(10^{-4}\), the plateau schedule, seed 42, and the checkpoint with the best validation loss. The baseline SIREN was trained once and reused. ReLU and Tanh use PyTorch's default initialization; SIREN uses the paper initialization.

```bash
python ablation_run.py
```

Committed results: `outputs/ablation/results.csv` and `outputs/ablation/ablation_plot.png`. R² below is on the same outside-only bands as the table above.

| Group | Variant | In-distribution R² | Near outside R² | Far outside R² |
|--------|---------|--------------------|-----------------|----------------|
| Activation | SIREN (baseline) | 1.000 | 0.674 | 0.489 |
| Activation | ReLU | 0.827 | -9.790 | -85.691 |
| Activation | Tanh | 0.386 | -1.862 | -5.319 |
| \(\omega_0\) | 1 | 1.000 | -1.562 | -0.708 |
| \(\omega_0\) | 10 | 1.000 | -0.159 | -0.255 |
| \(\omega_0\) | 30 (baseline) | 1.000 | 0.674 | 0.489 |
| \(\omega_0\) | 100 | 0.997 | 0.020 | -0.041 |
| Depth | 1 layer | 0.831 | 0.609 | 0.623 |
| Depth | 2 layers | 1.000 | 0.283 | 0.341 |
| Depth | 4 layers (baseline) | 1.000 | 0.674 | 0.489 |
| Depth | 6 layers | 1.000 | 0.547 | 0.520 |
| Sampling | Stratified (baseline) | 1.000 | 0.674 | 0.489 |
| Sampling | Uniform | 1.000 | 0.669 | 0.412 |
| Sampling | Grid | 1.000 | 0.667 | 0.469 |

The baseline is the best joint result: it fits the training window and has the highest R² just outside it. ReLU extrapolates as a straight line, so its outside R² collapses. Tanh flattens and, under this shared learning rate, does not fit the training window either. \(\omega_0 = 30\) is the only frequency in the sweep that both fits and extrapolates. One layer extrapolates decently but does not fit. Six layers are close to four and worse on the near band. Stratified, uniform, and grid sampling all fit; stratified is slightly better far away.

## Tests

```bash
pytest -q
```

Coverage includes sampling domain and strategies, dataset shapes, SIREN forward shapes and initialization, a short training run, evaluation metrics, and the ablation helpers (matched parameter counts and finite zone scores).

## Design choices

| Choice | Rationale |
|--------|-----------|
| Stratified sampling, 5k points | Coverage across ~10 periods in the training window without large gaps |
| MSE + Adam \(10^{-4}\) | Standard for clean regression / SIREN |
| `ReduceLROnPlateau` | Lower LR when validation stalls |
| Checkpoint best val loss | Prefer generalization over last epoch |
| \(\omega_0 = 30\) | Default from the SIREN paper |

## Limitations / failure cases

- **Period mismatch / multi-scale**: if the true \(f\) has very high-frequency content relative to \(\omega_0\) and width, fitting may need more capacity or longer training.
- **Extrapolation drifts**: outside-only R² falls from 0.67 just beyond the training window to 0.49 on the far bands. At \(x = 1100\) the prediction has the wrong sign (\(-0.32\) versus \(0.82\)).
- **This \(f\) is simple**: \(\sin x + 0.5\sin 2x\) has two frequencies. A target with sharper or less regular oscillations would be harder to fit, including inside the training window.
- **No period discovery report**: we do not estimate \(T\) explicitly (by design - not hardcoded, not returned).

## What I would improve

- A learned input warp or post-hoc period estimate, checked only at analysis time, to stop the far-region drift. The ablation shows that sine layers and \(\omega_0 = 30\) help, and that the far-band error remains.
- Hyperparameter search and early stopping with a held-out extrapolation split.
- Packaging (`pyproject.toml`) and a small inference-only CLI that loads a checkpoint and never imports `black_box`.
