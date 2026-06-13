"""Sequential Bayesian optimization loop for the Branin-Hoo benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .acquisition import get_acquisition
from .benchmarks import branin_hoo, make_branin_grid, sample_uniform
from .gp import gaussian_process_predict, optimize_hyperparameters
from .kernels import get_kernel


@dataclass(frozen=True)
class BOConfig:
    """Configuration for a sequential Bayesian optimization run."""

    kernel: str = "rbf"
    acquisition: str = "ei"
    n_initial: int = 10
    n_rounds: int = 20
    grid_size: int = 80
    seed: int = 0
    optimize_every: int = 5
    length_grid_size: int = 15
    signal_grid_size: int = 15
    noise_grid_size: int = 8
    device: str | None = None


def rmse(prediction, target):
    """Root mean squared error."""
    prediction = np.asarray(prediction, dtype=float).ravel()
    target = np.asarray(target, dtype=float).ravel()
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def choose_candidate(acquisition_name, acquisition_func, mu, sigma, y_best, rng, used_indices):
    """Choose the next grid candidate from posterior predictions."""
    n_candidates = len(mu)
    available = np.ones(n_candidates, dtype=bool)
    if used_indices:
        available[list(used_indices)] = False

    if not np.any(available):
        raise RuntimeError("No unused grid candidates remain.")

    if acquisition_name == "random":
        return int(rng.choice(np.flatnonzero(available)))

    scores = acquisition_func(mu, sigma, y_best)
    scores = np.asarray(scores, dtype=float).ravel()
    scores[~available] = -np.inf
    return int(np.argmax(scores))


def run_sequential_bo(config):
    """Run sequential BO and return metrics, selected points, and final state."""
    rng = np.random.default_rng(config.seed)
    kernel_func, kernel_label = get_kernel(config.kernel)
    acquisition_func = get_acquisition(config.acquisition)
    _, _, x_grid, true_values = make_branin_grid(config.grid_size)

    x_train = sample_uniform(config.n_initial, rng)
    y_train = branin_hoo(x_train)
    grid_best = float(np.min(true_values))
    used_grid_indices = set()
    records = []
    params = None

    for step in range(config.n_rounds + 1):
        should_optimize = params is None or (
            config.optimize_every > 0 and step % config.optimize_every == 0
        )
        if should_optimize:
            params = optimize_hyperparameters(
                x_train,
                y_train,
                config.kernel,
                length_grid=np.logspace(-1, 1, config.length_grid_size),
                sf_grid=np.logspace(-1, 1, config.signal_grid_size),
                sn_grid=np.logspace(-4, -1, config.noise_grid_size),
                device=config.device,
            )

        mu, sigma = gaussian_process_predict(x_train, y_train, x_grid, kernel_func, params)
        best_observed = float(np.min(y_train))
        records.append(
            {
                "step": step,
                "n_observations": len(x_train),
                "kernel": kernel_label,
                "acquisition": config.acquisition.upper(),
                "length_scale": params.length_scale,
                "sigma_f": params.sigma_f,
                "noise": params.noise,
                "best_observed": best_observed,
                "grid_best": grid_best,
                "simple_regret": best_observed - grid_best,
                "rmse": rmse(mu, true_values),
                "mean_sigma": float(np.mean(sigma)),
            }
        )

        if step == config.n_rounds:
            break

        next_idx = choose_candidate(
            config.acquisition,
            acquisition_func,
            mu,
            sigma,
            best_observed,
            rng,
            used_grid_indices,
        )
        used_grid_indices.add(next_idx)
        x_next = x_grid[next_idx]
        y_next = branin_hoo(x_next)
        x_train = np.vstack([x_train, x_next])
        y_train = np.append(y_train, y_next)

    return {
        "records": records,
        "x_train": x_train,
        "y_train": y_train,
        "x_grid": x_grid,
        "true_values": true_values,
        "kernel_label": kernel_label,
    }
