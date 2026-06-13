# Gaussian-Process Bayesian Optimization

This folder contains a from-scratch Gaussian Process experiment on the
Branin-Hoo function.

## What It Implements

- Branin-Hoo benchmark function
- RBF, Matern 1.5, and Rational Quadratic kernels
- Exact GP posterior mean and variance
- Grid-search hyperparameter selection by log marginal likelihood
- Expected Improvement, Probability of Improvement, Lower Confidence Bound, and
  random acquisition
- Multi-round sequential Bayesian optimization
- RMSE and uncertainty-error correlation reports

## Run

```bash
python -m bayesian_optimization.run_bo --kernel rbf --acquisition ei --n-initial 10 --n-rounds 20
```

This writes metrics to `outputs/sequential_bo/sequential_bo_metrics.csv` and a
convergence plot unless `--no-plot` is set. The output directory is intentionally
ignored by Git.

The original one-step result CSVs are kept as reference artifacts:

- `rmse_results.csv`
- `rmse_results_corr.csv`

## Smoke Test

```bash
python -m bayesian_optimization.run_bo --n-initial 5 --n-rounds 2 --grid-size 12 --no-plot
```

## Multi-Seed Comparison

```bash
python -m bayesian_optimization.compare_acquisitions --n-seeds 10 --n-rounds 30
```

This writes:

- `docs/results/sequential_bo_runs.csv`
- `docs/results/sequential_bo_summary.csv`
- `docs/assets/sequential_bo_regret.svg`

## Architecture

```text
benchmarks.py      Branin-Hoo objective and grid helpers
kernels.py         Kernel functions and registry
gp.py              Exact GP posterior and hyperparameter search
acquisition.py     Acquisition functions for minimization
sequential_bo.py   Multi-round optimization loop
run_bo.py          CLI runner
compare_acquisitions.py  Multi-seed acquisition comparison
```

The GP-only path can run without PyTorch. If PyTorch is installed, the
hyperparameter grid search uses a faster batched implementation.
