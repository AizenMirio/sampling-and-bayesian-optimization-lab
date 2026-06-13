"""Command-line runner for sequential Bayesian optimization."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .sequential_bo import BOConfig, run_sequential_bo


def write_records(records, path):
    """Write metric records to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def plot_convergence(records, path):
    """Plot best observed value and model RMSE over BO rounds."""
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    steps = [record["step"] for record in records]
    best_values = [record["best_observed"] for record in records]
    rmses = [record["rmse"] for record in records]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(steps, best_values, marker="o")
    axes[0].set_title("Best observed value")
    axes[0].set_xlabel("BO round")
    axes[0].set_ylabel("f(x), lower is better")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, rmses, marker="o", color="tab:orange")
    axes[1].set_title("GP approximation RMSE")
    axes[1].set_xlabel("BO round")
    axes[1].set_ylabel("RMSE over grid")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", default="rbf", choices=["rbf", "matern", "rational_quadratic"])
    parser.add_argument("--acquisition", default="ei", choices=["ei", "pi", "lcb", "random"])
    parser.add_argument("--n-initial", type=int, default=10)
    parser.add_argument("--n-rounds", type=int, default=20)
    parser.add_argument("--grid-size", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--optimize-every", type=int, default=5)
    parser.add_argument("--length-grid-size", type=int, default=15)
    parser.add_argument("--signal-grid-size", type=int, default=15)
    parser.add_argument("--noise-grid-size", type=int, default=8)
    parser.add_argument("--device", default=None, help="Torch device for hyperparameter search.")
    parser.add_argument("--output-dir", default="outputs/sequential_bo")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = BOConfig(
        kernel=args.kernel,
        acquisition=args.acquisition,
        n_initial=args.n_initial,
        n_rounds=args.n_rounds,
        grid_size=args.grid_size,
        seed=args.seed,
        optimize_every=args.optimize_every,
        length_grid_size=args.length_grid_size,
        signal_grid_size=args.signal_grid_size,
        noise_grid_size=args.noise_grid_size,
        device=args.device,
    )
    result = run_sequential_bo(config)
    output_dir = Path(args.output_dir)
    metrics_path = output_dir / "sequential_bo_metrics.csv"
    write_records(result["records"], metrics_path)

    if not args.no_plot:
        plot_convergence(result["records"], output_dir / "convergence.png")

    final = result["records"][-1]
    print(f"Wrote metrics: {metrics_path}")
    print(
        "Final "
        f"best_observed={final['best_observed']:.4f}, "
        f"simple_regret={final['simple_regret']:.4f}, "
        f"rmse={final['rmse']:.4f}"
    )


if __name__ == "__main__":
    main()
