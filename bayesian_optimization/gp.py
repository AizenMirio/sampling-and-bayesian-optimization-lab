"""Exact Gaussian-process regression utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # Torch is optional for the GP-only path.
    torch = None


@dataclass(frozen=True)
class GPParams:
    """Kernel hyperparameters for an exact GP."""

    length_scale: float = 1.0
    sigma_f: float = 1.0
    noise: float = 1e-2


def log_marginal_likelihood(x_train, y_train, kernel_func, params):
    """Compute the exact GP log marginal likelihood."""
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    k_train = kernel_func(
        x_train,
        x_train,
        length_scale=params.length_scale,
        sigma_f=params.sigma_f,
    )
    k_train = k_train + params.noise**2 * np.eye(len(x_train))
    try:
        chol = np.linalg.cholesky(k_train + 1e-10 * np.eye(len(x_train)))
    except np.linalg.LinAlgError:
        return -np.inf
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y_train))
    logdet = 2.0 * np.sum(np.log(np.diag(chol)))
    return -0.5 * y_train @ alpha - 0.5 * logdet - 0.5 * len(x_train) * np.log(2 * np.pi)


def optimize_hyperparameters(
    x_train,
    y_train,
    kernel_name,
    length_grid=None,
    sf_grid=None,
    sn_grid=None,
    device=None,
):
    """Grid-search GP hyperparameters by log marginal likelihood."""
    if length_grid is None:
        length_grid = np.logspace(-1, 1, 15)
    if sf_grid is None:
        sf_grid = np.logspace(-1, 1, 15)
    if sn_grid is None:
        sn_grid = np.logspace(-4, -1, 8)
    if torch is None:
        return _optimize_hyperparameters_numpy(
            x_train,
            y_train,
            kernel_name,
            length_grid,
            sf_grid,
            sn_grid,
        )
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    x_tensor = torch.as_tensor(x_train, dtype=torch.double, device=device)
    y_tensor = torch.as_tensor(y_train, dtype=torch.double, device=device).view(-1, 1)
    n_train = x_tensor.shape[0]
    eye = torch.eye(n_train, dtype=torch.double, device=device)

    sq_dists = (
        x_tensor.pow(2).sum(1, keepdim=True)
        + x_tensor.pow(2).sum(1)[None, :]
        - 2.0 * x_tensor @ x_tensor.T
    )

    def kernel_base(ell, sf):
        if kernel_name == "rbf":
            return sf**2 * torch.exp(-0.5 * sq_dists / ell**2)
        if kernel_name == "matern":
            dists = torch.sqrt(torch.clamp(sq_dists, min=0.0) + 1e-12)
            scaled = torch.sqrt(torch.tensor(3.0, dtype=torch.double, device=device)) * dists / ell
            return sf**2 * (1.0 + scaled) * torch.exp(-scaled)
        if kernel_name == "rational_quadratic":
            return sf**2 * torch.pow(1.0 + sq_dists / (2.0 * ell**2), -1.0)
        raise ValueError(f"Unknown kernel: {kernel_name}")

    best_ll = -torch.inf
    best_params = None

    for ell in length_grid:
        ell_tensor = torch.tensor(float(ell), dtype=torch.double, device=device)
        for sf in sf_grid:
            sf_tensor = torch.tensor(float(sf), dtype=torch.double, device=device)
            k_base = kernel_base(ell_tensor, sf_tensor)
            noise_vars = torch.as_tensor(sn_grid**2, dtype=torch.double, device=device).view(-1, 1, 1)
            k_all = k_base.unsqueeze(0) + noise_vars * eye + 1e-6 * eye
            try:
                chol_all = torch.linalg.cholesky(k_all)
            except RuntimeError:
                continue

            logdet = 2.0 * torch.log(torch.diagonal(chol_all, dim1=-2, dim2=-1)).sum(-1)
            alpha = torch.cholesky_solve(y_tensor.expand(chol_all.shape[0], n_train, 1), chol_all)
            quad = (y_tensor.view(1, n_train, 1) * alpha).sum((-2, -1))
            ll = -0.5 * quad - 0.5 * logdet - 0.5 * n_train * np.log(2 * np.pi)
            max_ll, idx = ll.max(dim=0)
            if max_ll > best_ll:
                best_ll = max_ll
                best_params = GPParams(float(ell), float(sf), float(sn_grid[idx]))

    return best_params or GPParams()


def _optimize_hyperparameters_numpy(x_train, y_train, kernel_name, length_grid, sf_grid, sn_grid):
    """NumPy fallback for machines without PyTorch."""
    from .kernels import get_kernel

    kernel_func, _ = get_kernel(kernel_name)
    best_ll = -np.inf
    best_params = None
    for length_scale in length_grid:
        for sigma_f in sf_grid:
            for noise in sn_grid:
                params = GPParams(float(length_scale), float(sigma_f), float(noise))
                ll = log_marginal_likelihood(x_train, y_train, kernel_func, params)
                if ll > best_ll:
                    best_ll = ll
                    best_params = params
    return best_params or GPParams()


def gaussian_process_predict(x_train, y_train, x_test, kernel_func, params):
    """Predict GP posterior mean and standard deviation at test points."""
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    x_test = np.asarray(x_test, dtype=float)

    k_train = kernel_func(
        x_train,
        x_train,
        length_scale=params.length_scale,
        sigma_f=params.sigma_f,
    )
    k_train = k_train + params.noise**2 * np.eye(len(x_train))
    k_cross = kernel_func(
        x_train,
        x_test,
        length_scale=params.length_scale,
        sigma_f=params.sigma_f,
    )
    chol = np.linalg.cholesky(k_train + 1e-10 * np.eye(len(x_train)))
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y_train))
    mu = k_cross.T @ alpha
    v = np.linalg.solve(chol, k_cross)
    # For the stationary kernels used here, k(x, x) is sigma_f^2 for every x.
    # Avoid materializing the full test-test covariance matrix.
    prior_var = np.full(len(x_test), params.sigma_f**2, dtype=float)
    var = prior_var - np.sum(v**2, axis=0)
    return mu, np.sqrt(np.maximum(var, 1e-12))
