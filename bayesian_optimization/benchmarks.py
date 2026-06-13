"""Benchmark functions and grids for Bayesian optimization experiments."""

from __future__ import annotations

import numpy as np


BRANIN_DOMAIN = np.array([[-5.0, 10.0], [0.0, 15.0]], dtype=float)


def branin_hoo(x):
    """Compute the Branin-Hoo function for one point or an array of points."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    x1, x2 = x[:, 0], x[:, 1]
    y = (
        (x2 - 5.1 / (4.0 * np.pi**2) * x1**2 + 5.0 / np.pi * x1 - 6.0) ** 2
        + 10.0 * (1.0 - 1.0 / (8.0 * np.pi)) * np.cos(x1)
        + 10.0
    )
    return y if len(y) > 1 else y.item()


def make_branin_grid(grid_size=100):
    """Return a dense Branin-Hoo evaluation grid."""
    x1_test = np.linspace(BRANIN_DOMAIN[0, 0], BRANIN_DOMAIN[0, 1], grid_size)
    x2_test = np.linspace(BRANIN_DOMAIN[1, 0], BRANIN_DOMAIN[1, 1], grid_size)
    x1_grid, x2_grid = np.meshgrid(x1_test, x2_test)
    x_test = np.column_stack([x1_grid.ravel(), x2_grid.ravel()])
    true_values = branin_hoo(x_test).reshape(-1)
    return x1_grid, x2_grid, x_test, true_values


def sample_uniform(n_samples, rng, domain=BRANIN_DOMAIN):
    """Draw uniform random samples from a rectangular 2D domain."""
    low = domain[:, 0]
    high = domain[:, 1]
    return rng.uniform(low=low, high=high, size=(n_samples, domain.shape[0]))

