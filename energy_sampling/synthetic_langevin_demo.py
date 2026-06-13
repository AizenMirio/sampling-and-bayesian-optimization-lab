"""Reproducible Langevin sampling demo on a 2D double-well energy."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def double_well_energy(x):
    """Two-mode energy landscape with minima near x1=-1 and x1=1."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    x1 = x[:, 0]
    x2 = x[:, 1]
    return 0.25 * (x1**2 - 1.0) ** 2 + 0.5 * x2**2


def double_well_grad(x):
    """Gradient of double_well_energy."""
    x = np.asarray(x, dtype=float)
    grad = np.empty_like(x, dtype=float)
    grad[:, 0] = x[:, 0] * (x[:, 0] ** 2 - 1.0)
    grad[:, 1] = x[:, 1]
    return grad


def _log_proposal_density(x_to, mean, step_size):
    diff = x_to - mean
    return -np.sum(diff**2, axis=1) / (2.0 * step_size)


def ula_sample(n_chains=512, n_steps=500, step_size=0.02, seed=0):
    """Sample with the Unadjusted Langevin Algorithm."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_chains, 2))
    noise_scale = np.sqrt(step_size)
    for _ in range(n_steps):
        x = x - 0.5 * step_size * double_well_grad(x) + noise_scale * rng.normal(size=x.shape)
    return x


def mala_sample(n_chains=512, n_steps=500, step_size=0.02, seed=0):
    """Sample with Metropolis-Adjusted Langevin Algorithm."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_chains, 2))
    noise_scale = np.sqrt(step_size)
    accepted = 0
    total = 0

    for _ in range(n_steps):
        grad_x = double_well_grad(x)
        proposal_mean = x - 0.5 * step_size * grad_x
        proposal = proposal_mean + noise_scale * rng.normal(size=x.shape)

        grad_prop = double_well_grad(proposal)
        reverse_mean = proposal - 0.5 * step_size * grad_prop

        log_target_current = -double_well_energy(x)
        log_target_proposal = -double_well_energy(proposal)
        log_q_forward = _log_proposal_density(proposal, proposal_mean, step_size)
        log_q_backward = _log_proposal_density(x, reverse_mean, step_size)
        log_accept_ratio = (
            log_target_proposal
            - log_target_current
            + log_q_backward
            - log_q_forward
        )

        accept = np.log(rng.uniform(size=n_chains)) < np.minimum(0.0, log_accept_ratio)
        x[accept] = proposal[accept]
        accepted += int(np.sum(accept))
        total += n_chains

    return x, accepted / total


def summarize_samples(samples):
    """Return compact diagnostics for samples from the double-well target."""
    energies = double_well_energy(samples)
    return {
        "mean_x1": float(np.mean(samples[:, 0])),
        "mean_x2": float(np.mean(samples[:, 1])),
        "std_x1": float(np.std(samples[:, 0])),
        "std_x2": float(np.std(samples[:, 1])),
        "left_mode_fraction": float(np.mean(samples[:, 0] < 0.0)),
        "right_mode_fraction": float(np.mean(samples[:, 0] >= 0.0)),
        "mean_energy": float(np.mean(energies)),
    }


def write_summary(rows, output_path):
    """Write sampler diagnostics to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_samples(sample_sets, output_path):
    """Save a scatter plot for sampler comparison."""
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(sample_sets), figsize=(5 * len(sample_sets), 4))
    if len(sample_sets) == 1:
        axes = [axes]
    for axis, (name, samples) in zip(axes, sample_sets.items()):
        axis.scatter(samples[:, 0], samples[:, 1], s=8, alpha=0.35)
        axis.set_title(name)
        axis.set_xlabel("x1")
        axis.set_ylabel("x2")
        axis.set_xlim(-3, 3)
        axis.set_ylim(-3, 3)
        axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-chains", type=int, default=512)
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--step-size", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/synthetic_energy")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    ula_samples = ula_sample(args.n_chains, args.n_steps, args.step_size, args.seed)
    mala_samples, acceptance_rate = mala_sample(
        args.n_chains,
        args.n_steps,
        args.step_size,
        args.seed,
    )

    rows = []
    for name, samples in [("ULA", ula_samples), ("MALA", mala_samples)]:
        row = {"sampler": name, **summarize_samples(samples)}
        row["acceptance_rate"] = "" if name == "ULA" else f"{acceptance_rate:.6f}"
        rows.append(row)

    summary_path = output_dir / "synthetic_langevin_summary.csv"
    write_summary(rows, summary_path)
    if not args.no_plot:
        plot_samples({"ULA": ula_samples, "MALA": mala_samples}, output_dir / "synthetic_langevin_samples.png")

    print(f"Wrote summary: {summary_path}")
    print(f"MALA acceptance_rate={acceptance_rate:.4f}")


if __name__ == "__main__":
    main()

