# Reproducibility Notes

This repo has two execution paths:

1. A lightweight local path that runs without external weights or datasets.
2. An optional neural-energy reference path that requires local model weights and
   tensor data.

## Lightweight Path

Install the core dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the synthetic Langevin demo:

```bash
python -m energy_sampling.synthetic_langevin_demo --no-plot
```

Run a short Bayesian optimization smoke test:

```bash
python -m bayesian_optimization.run_bo --n-initial 5 --n-rounds 2 --grid-size 12 --no-plot
```

Run the stdlib test suite:

```bash
python -m unittest discover
```

## Cleanup Experiment Commands

Regenerate the multi-seed Bayesian optimization comparison:

```bash
python -m bayesian_optimization.compare_acquisitions --n-seeds 10 --n-rounds 30
```

Regenerate the synthetic energy step-size sweep:

```bash
python -m energy_sampling.compare_step_sizes --n-seeds 5
```

These commands write CSV summaries to `docs/results/` and SVG plots to
`docs/assets/`.

## Optional Neural-Energy Reference Path

The high-dimensional neural-energy code requires PyTorch plus local weights and
data.

Install optional dependencies:

```bash
python -m pip install -r requirements-neural.txt
```

Evaluate a saved neural energy regressor:

```bash
python -m energy_sampling.neural_energy_model --weights path/to/model.pth --dataset path/to/data.pt
```

Sample from the saved neural energy regressor:

```bash
python -m energy_sampling.neural_langevin_samplers --weights path/to/model.pth
```

## Notes

- Original course reports, generated plot dumps, and model weights are not
  committed.
- The checked-in synthetic energy demo is the easiest way to inspect ULA/MALA
  behavior without external artifacts.
- The original one-step GP result CSVs are preserved as reference artifacts.
