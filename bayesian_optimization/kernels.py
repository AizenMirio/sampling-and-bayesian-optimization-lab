"""Kernel functions used by the Gaussian-process model."""

from __future__ import annotations

import numpy as np


def ensure_2d(x):
    """Convert 1D arrays to column vectors and preserve 2D arrays."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    return x


def squared_distances(x1, x2):
    """Pairwise squared Euclidean distances."""
    x1, x2 = ensure_2d(x1), ensure_2d(x2)
    return (
        np.sum(x1**2, axis=1)[:, None]
        + np.sum(x2**2, axis=1)[None, :]
        - 2.0 * x1 @ x2.T
    )


def rbf_kernel(x1, x2, length_scale=1.0, sigma_f=1.0):
    """Isotropic squared-exponential kernel."""
    sqdist = squared_distances(x1, x2)
    return sigma_f**2 * np.exp(-0.5 * sqdist / length_scale**2)


def matern_kernel(x1, x2, length_scale=1.0, sigma_f=1.0, nu=1.5):
    """Matern kernel with nu=1.5."""
    if nu != 1.5:
        raise ValueError("Only nu=1.5 is supported.")
    d = np.sqrt(np.maximum(squared_distances(x1, x2), 0.0))
    scaled = np.sqrt(3.0) * d / length_scale
    return sigma_f**2 * (1.0 + scaled) * np.exp(-scaled)


def rational_quadratic_kernel(x1, x2, length_scale=1.0, sigma_f=1.0, alpha=1.0):
    """Rational Quadratic kernel."""
    sqdist = squared_distances(x1, x2)
    return sigma_f**2 * (1.0 + sqdist / (2.0 * alpha * length_scale**2)) ** (-alpha)


KERNELS = {
    "rbf": (rbf_kernel, "RBF"),
    "matern": (matern_kernel, "Matern 1.5"),
    "rational_quadratic": (rational_quadratic_kernel, "Rational Quadratic"),
}


def get_kernel(name):
    """Resolve a kernel name to its callable and display label."""
    try:
        return KERNELS[name]
    except KeyError as exc:
        names = ", ".join(sorted(KERNELS))
        raise ValueError(f"Unknown kernel '{name}'. Choose one of: {names}") from exc

