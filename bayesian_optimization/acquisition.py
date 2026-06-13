"""Acquisition functions for minimization-oriented Bayesian optimization."""

from __future__ import annotations

import math

import numpy as np


def normal_pdf(z):
    """Standard normal probability density."""
    return np.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)


def normal_cdf(z):
    """Standard normal CDF without requiring SciPy."""
    z = np.asarray(z, dtype=float)
    erf_values = np.vectorize(math.erf)(z / math.sqrt(2.0))
    return 0.5 * (1.0 + erf_values)


def expected_improvement(mu, sigma, y_best, xi=0.01):
    """Expected Improvement for minimization."""
    sigma = np.maximum(sigma, 1e-12)
    improvement = y_best - mu - xi
    z = improvement / sigma
    return improvement * normal_cdf(z) + sigma * normal_pdf(z)


def probability_of_improvement(mu, sigma, y_best, xi=0.01):
    """Probability of Improvement for minimization."""
    sigma = np.maximum(sigma, 1e-12)
    return normal_cdf((y_best - mu - xi) / sigma)


def lower_confidence_bound(mu, sigma, beta=2.0):
    """Return an acquisition score based on the lower confidence bound."""
    return -(mu - beta * sigma)


ACQUISITIONS = {
    "ei": expected_improvement,
    "pi": probability_of_improvement,
    "lcb": lower_confidence_bound,
    "random": None,
}


def get_acquisition(name):
    """Resolve an acquisition name."""
    try:
        return ACQUISITIONS[name]
    except KeyError as exc:
        names = ", ".join(sorted(ACQUISITIONS))
        raise ValueError(f"Unknown acquisition '{name}'. Choose one of: {names}") from exc

