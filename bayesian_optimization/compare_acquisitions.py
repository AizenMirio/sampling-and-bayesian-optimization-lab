"""Compare acquisition functions across multiple BO seeds."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from .sequential_bo import BOConfig, run_sequential_bo


COLORS = {
    "EI": "#2563eb",
    "PI": "#16a34a",
    "LCB": "#dc2626",
    "RANDOM": "#6b7280",
}


def mean(values):
    return sum(values) / len(values)


def std(values):
    if len(values) <= 1:
        return 0.0
    mu = mean(values)
    return (sum((value - mu) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def auc_simple_regret(records):
    """Area under the simple-regret curve with unit step spacing."""
    regrets = [record["simple_regret"] for record in records]
    if len(regrets) <= 1:
        return regrets[0] if regrets else 0.0
    area = 0.0
    for left, right in zip(regrets, regrets[1:]):
        area += 0.5 * (left + right)
    return area


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_runs(all_records):
    grouped = defaultdict(list)
    for run_key, records in all_records.items():
        acquisition = run_key[0]
        final = records[-1]
        grouped[acquisition].append(
            {
                "final_best_observed": final["best_observed"],
                "final_simple_regret": final["simple_regret"],
                "auc_simple_regret": auc_simple_regret(records),
                "final_rmse": final["rmse"],
            }
        )

    rows = []
    for acquisition, values in sorted(grouped.items()):
        rows.append(
            {
                "acquisition": acquisition.upper(),
                "n_runs": len(values),
                "final_best_observed_mean": mean([v["final_best_observed"] for v in values]),
                "final_best_observed_std": std([v["final_best_observed"] for v in values]),
                "final_simple_regret_mean": mean([v["final_simple_regret"] for v in values]),
                "final_simple_regret_std": std([v["final_simple_regret"] for v in values]),
                "auc_simple_regret_mean": mean([v["auc_simple_regret"] for v in values]),
                "auc_simple_regret_std": std([v["auc_simple_regret"] for v in values]),
                "final_rmse_mean": mean([v["final_rmse"] for v in values]),
                "final_rmse_std": std([v["final_rmse"] for v in values]),
            }
        )
    return rows


def mean_regret_by_step(all_records):
    grouped = defaultdict(lambda: defaultdict(list))
    for (acquisition, _seed), records in all_records.items():
        for record in records:
            grouped[acquisition.upper()][record["step"]].append(record["simple_regret"])
    return {
        acquisition: {step: mean(values) for step, values in steps.items()}
        for acquisition, steps in grouped.items()
    }


def _scale(value, src_min, src_max, dst_min, dst_max):
    if src_max == src_min:
        return 0.5 * (dst_min + dst_max)
    return dst_min + (value - src_min) * (dst_max - dst_min) / (src_max - src_min)


def write_regret_svg(mean_curves, path):
    """Write a simple SVG line plot without external plotting dependencies."""
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 760, 440
    margin_left, margin_right, margin_top, margin_bottom = 72, 28, 34, 62
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    all_steps = sorted({step for curve in mean_curves.values() for step in curve})
    all_values = [value for curve in mean_curves.values() for value in curve.values()]
    x_min, x_max = min(all_steps), max(all_steps)
    y_min, y_max = 0.0, max(all_values) * 1.05

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="22" text-anchor="middle" font-family="Arial" font-size="16" font-weight="700">Sequential BO: Mean Simple Regret</text>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#111827"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#111827"/>',
        f'<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-family="Arial" font-size="12">BO round</text>',
        f'<text transform="translate(18 {height / 2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="12">Simple regret, lower is better</text>',
    ]

    for tick in range(0, x_max + 1, max(1, x_max // 5)):
        x = _scale(tick, x_min, x_max, margin_left, margin_left + plot_w)
        lines.append(f'<line x1="{x:.1f}" y1="{height - margin_bottom}" x2="{x:.1f}" y2="{height - margin_bottom + 5}" stroke="#111827"/>')
        lines.append(f'<text x="{x:.1f}" y="{height - margin_bottom + 20}" text-anchor="middle" font-family="Arial" font-size="11">{tick}</text>')

    for i in range(6):
        value = y_min + i * (y_max - y_min) / 5
        y = _scale(value, y_min, y_max, height - margin_bottom, margin_top)
        lines.append(f'<line x1="{margin_left - 5}" y1="{y:.1f}" x2="{margin_left}" y2="{y:.1f}" stroke="#111827"/>')
        lines.append(f'<text x="{margin_left - 9}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{value:.1f}</text>')
        lines.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#e5e7eb"/>')

    legend_x = width - 145
    legend_y = margin_top + 10
    for idx, (acquisition, curve) in enumerate(sorted(mean_curves.items())):
        color = COLORS.get(acquisition.upper(), "#111827")
        points = []
        for step in sorted(curve):
            x = _scale(step, x_min, x_max, margin_left, margin_left + plot_w)
            y = _scale(curve[step], y_min, y_max, height - margin_bottom, margin_top)
            points.append(f"{x:.1f},{y:.1f}")
        lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.4"/>')
        y_legend = legend_y + idx * 22
        lines.append(f'<line x1="{legend_x}" y1="{y_legend}" x2="{legend_x + 22}" y2="{y_legend}" stroke="{color}" stroke-width="2.4"/>')
        lines.append(f'<text x="{legend_x + 28}" y="{y_legend + 4}" font-family="Arial" font-size="12">{acquisition}</text>')

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", default="rbf", choices=["rbf", "matern", "rational_quadratic"])
    parser.add_argument("--acquisitions", nargs="+", default=["ei", "pi", "lcb", "random"])
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--n-initial", type=int, default=10)
    parser.add_argument("--n-rounds", type=int, default=30)
    parser.add_argument("--grid-size", type=int, default=60)
    parser.add_argument("--optimize-every", type=int, default=0)
    parser.add_argument("--length-grid-size", type=int, default=9)
    parser.add_argument("--signal-grid-size", type=int, default=9)
    parser.add_argument("--noise-grid-size", type=int, default=5)
    parser.add_argument("--output-dir", default="docs/results")
    parser.add_argument("--plot-path", default="docs/assets/sequential_bo_regret.svg")
    return parser.parse_args()


def main():
    args = parse_args()
    all_records = {}
    flat_records = []

    for acquisition in args.acquisitions:
        for seed in range(args.seed_offset, args.seed_offset + args.n_seeds):
            config = BOConfig(
                kernel=args.kernel,
                acquisition=acquisition,
                n_initial=args.n_initial,
                n_rounds=args.n_rounds,
                grid_size=args.grid_size,
                seed=seed,
                optimize_every=args.optimize_every,
                length_grid_size=args.length_grid_size,
                signal_grid_size=args.signal_grid_size,
                noise_grid_size=args.noise_grid_size,
            )
            result = run_sequential_bo(config)
            records = result["records"]
            all_records[(acquisition, seed)] = records
            for record in records:
                flat_records.append({"seed": seed, **record})
            final = records[-1]
            print(
                f"{acquisition.upper():>6s} seed={seed:02d} "
                f"best={final['best_observed']:.4f} regret={final['simple_regret']:.4f} "
                f"rmse={final['rmse']:.4f}"
            )

    output_dir = Path(args.output_dir)
    write_csv(flat_records, output_dir / "sequential_bo_runs.csv")
    summary_rows = summarize_runs(all_records)
    write_csv(summary_rows, output_dir / "sequential_bo_summary.csv")
    write_regret_svg(mean_regret_by_step(all_records), Path(args.plot_path))
    print(f"Wrote run metrics: {output_dir / 'sequential_bo_runs.csv'}")
    print(f"Wrote summary: {output_dir / 'sequential_bo_summary.csv'}")
    print(f"Wrote plot: {args.plot_path}")


if __name__ == "__main__":
    main()

