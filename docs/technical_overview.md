# Technical Overview

This repository focuses on two connected themes: sampling from an energy
function and using Gaussian Process uncertainty to guide optimization.

## 1. Energy-Based Sampling

An energy-based model assigns a scalar energy `E(x)` to an input `x`. Lower
energy means the model considers that point more likely. Unlike a normal
classifier, the model does not directly return a normalized probability.

The sampling problem is:

```text
Given only E(x), generate samples that concentrate in low-energy regions.
```

The repository includes Langevin-style sampler code and a small synthetic demo.

The public repo contains two versions of this idea:

- a fully reproducible 2D double-well energy demo,
- an implementation reference for the original neural-energy experiment.

### ULA

ULA stands for Unadjusted Langevin Algorithm. Each step moves the current sample
in the direction that reduces energy, then adds random Gaussian noise:

```text
x_next = x - 0.5 * step_size * grad_x E(x) + sqrt(step_size) * noise
```

The gradient term pulls samples toward low-energy regions. The noise term keeps
the sampler from collapsing into a single local minimum.

ULA is fast, but approximate. Because it accepts every proposal, a large step
size can bias the chain or push samples into poor regions.

### MALA

MALA stands for Metropolis-Adjusted Langevin Algorithm. It proposes a Langevin
step like ULA, but then applies a Metropolis-Hastings accept/reject correction.

That correction makes the sampler more principled, but also more expensive:

- it evaluates the energy at the current and proposed points,
- it computes gradients for the reverse proposal,
- it may reject a proposed move.

In the original run, MALA took about 2.5x as long as ULA, but produced a more
concentrated t-SNE projection.

## 2. Gaussian-Process Bayesian Optimization

The second half studies Gaussian Process regression on the Branin-Hoo function,
a standard two-dimensional benchmark for optimization.

A GP is useful because it predicts both:

- a mean estimate `mu(x)`,
- an uncertainty estimate `sigma(x)`.

That uncertainty lets us ask:

```text
Where should we sample next if function evaluations are expensive?
```

### Kernels

The kernel defines what kinds of functions the GP expects before seeing data.

- **RBF:** very smooth functions.
- **Matern 1.5:** rougher functions with less smoothness.
- **Rational Quadratic:** behaves like a mixture of RBF kernels at different
  length scales.

The experiment uses grid search over length scale, signal variance, and noise to
choose kernel hyperparameters by log marginal likelihood.

### Acquisition Functions

Acquisition functions turn GP predictions into a decision rule.

- **Expected Improvement:** favors points that are likely to improve and have
  enough uncertainty to be worth exploring.
- **Probability of Improvement:** favors points with high chance of beating the
  current best.
- **Random:** a baseline that ignores the model.

The original experiment added one acquisition-selected point after the initial
design. The repo also includes a sequential loop
that repeatedly fits the GP, selects a candidate point, evaluates the objective,
and updates the dataset.

## What This Demonstrates

This repo is useful because it shows you can:

- implement model-based sampling from gradients,
- reason about speed vs correctness in MCMC,
- implement GP posterior prediction from kernels,
- compare acquisition strategies,
- analyze prediction error and uncertainty calibration,
- turn experiment outputs into clear tables and interpretation.

## Limitations

- The neural-energy sampler depends on local weights/data that are not included,
  but the 2D synthetic demo is fully runnable.
- The original CSVs are from a one-step acquisition comparison; the new runner
  supports multi-round BO.
- EI and PI now use a standard normal CDF/PDF implementation based on `erf`
  instead of the earlier logistic approximation.
- Generated plots need to remain curated; committing every plot makes the repo
  noisy.
