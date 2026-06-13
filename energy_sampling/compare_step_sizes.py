"""Compare ULA and MALA across step sizes on the synthetic double-well energy."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from .synthetic_langevin_demo import mala_sample, summarize_samples, ula_sample


COLORS = {"ULA": "#2563eb", "MALA": "#dc2626"}


def mean(values):
    return sum(values) / len(values)


def std(values):
    if len(values) <= 1:
        return 0.0
    mu = mean(values)
    return (sum((value - mu) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["sampler"], row["step_size"])].append(row)

    summary_rows = []
    for (sampler, step_size), values in sorted(grouped.items(), key=lambda item: (float(item[0][1]), item[0][0])):
        energies = [row["mean_energy"] for row in values]
        imbalances = [row["mode_imbalance"] for row in values]
        acceptance_rates = [row["acceptance_rate"] for row in values if row["acceptance_rate"] != ""]
        summary_rows.append(
            {
                "sampler": sampler,
                "step_size": step_size,
                "n_runs": len(values),
                "mean_energy_mean": mean(energies),
                "mean_energy_std": std(energies),
                "mode_imbalance_mean": mean(imbalances),
                "mode_imbalance_std": std(imbalances),
                "acceptance_rate_mean": "" if not acceptance_rates else mean(acceptance_rates),
                "acceptance_rate_std": "" if not acceptance_rates else std(acceptance_rates),
            }
        )
    return summary_rows


def _scale(value, src_min, src_max, dst_min, dst_max):
    if src_max == src_min:
        return 0.5 * (dst_min + dst_max)
    return dst_min + (value - src_min) * (dst_max - dst_min) / (src_max - src_min)


def write_energy_svg(summary_rows, path):
    """Write a simple SVG plot of mean energy vs step size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 760, 440
    margin_left, margin_right, margin_top, margin_bottom = 72, 28, 34, 62
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    step_sizes = sorted({float(row["step_size"]) for row in summary_rows})
    y_values = [float(row["mean_energy_mean"]) for row in summary_rows]
    y_min, y_max = 0.0, max(y_values) * 1.08
    x_min, x_max = min(step_sizes), max(step_sizes)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="22" text-anchor="middle" font-family="Arial" font-size="16" font-weight="700">Synthetic Energy Sampling: Mean Energy</text>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#111827"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#111827"/>',
        f'<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-family="Arial" font-size="12">Step size</text>',
        f'<text transform="translate(18 {height / 2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="12">Mean energy, lower is better</text>',
    ]

    for step_size in step_sizes:
        x = _scale(step_size, x_min, x_max, margin_left, margin_left + plot_w)
        lines.append(f'<line x1="{x:.1f}" y1="{height - margin_bottom}" x2="{x:.1f}" y2="{height - margin_bottom + 5}" stroke="#111827"/>')
        lines.append(f'<text x="{x:.1f}" y="{height - margin_bottom + 20}" text-anchor="middle" font-family="Arial" font-size="11">{step_size:g}</text>')

    for i in range(6):
        value = y_min + i * (y_max - y_min) / 5
        y = _scale(value, y_min, y_max, height - margin_bottom, margin_top)
        lines.append(f'<line x1="{margin_left - 5}" y1="{y:.1f}" x2="{margin_left}" y2="{y:.1f}" stroke="#111827"/>')
        lines.append(f'<text x="{margin_left - 9}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{value:.2f}</text>')
        lines.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#e5e7eb"/>')

    for idx, sampler in enumerate(["ULA", "MALA"]):
        color = COLORS[sampler]
        rows = [row for row in summary_rows if row["sampler"] == sampler]
        points = []
        for row in rows:
            x = _scale(float(row["step_size"]), x_min, x_max, margin_left, margin_left + plot_w)
            y = _scale(float(row["mean_energy_mean"]), y_min, y_max, height - margin_bottom, margin_top)
            points.append(f"{x:.1f},{y:.1f}")
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}"/>')
        lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.4"/>')
        legend_y = margin_top + 12 + idx * 22
        legend_x = width - 135
        lines.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 22}" y2="{legend_y}" stroke="{color}" stroke-width="2.4"/>')
        lines.append(f'<text x="{legend_x + 28}" y="{legend_y + 4}" font-family="Arial" font-size="12">{sampler}</text>')

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step-sizes", nargs="+", type=float, default=[0.005, 0.02, 0.05, 0.1, 0.2])
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--n-chains", type=int, default=512)
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--output-dir", default="docs/results")
    parser.add_argument("--plot-path", default="docs/assets/synthetic_energy_step_sweep.svg")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    for step_size in args.step_sizes:
        for seed in range(args.seed_offset, args.seed_offset + args.n_seeds):
            ula_samples = ula_sample(args.n_chains, args.n_steps, step_size, seed)
            mala_samples, acceptance_rate = mala_sample(args.n_chains, args.n_steps, step_size, seed)
            for sampler, samples, rate in [
                ("ULA", ula_samples, ""),
                ("MALA", mala_samples, acceptance_rate),
            ]:
                summary = summarize_samples(samples)
                row = {
                    "sampler": sampler,
                    "seed": seed,
                    "step_size": step_size,
                    **summary,
                    "mode_imbalance": abs(summary["left_mode_fraction"] - 0.5),
                    "acceptance_rate": rate,
                }
                rows.append(row)
            print(f"step_size={step_size:g} seed={seed:02d} mala_acceptance={acceptance_rate:.4f}")

    output_dir = Path(args.output_dir)
    write_csv(rows, output_dir / "synthetic_energy_step_sweep_runs.csv")
    summary_rows = summarize(rows)
    write_csv(summary_rows, output_dir / "synthetic_energy_step_sweep_summary.csv")
    write_energy_svg(summary_rows, Path(args.plot_path))
    print(f"Wrote run metrics: {output_dir / 'synthetic_energy_step_sweep_runs.csv'}")
    print(f"Wrote summary: {output_dir / 'synthetic_energy_step_sweep_summary.csv'}")
    print(f"Wrote plot: {args.plot_path}")


if __name__ == "__main__":
    main()

