#!/usr/bin/env python3
"""Regenerate ICC-2025-style plots from generated CSV tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate_icc2025_style_results import (
    DETECTOR_COLORS,
    DETECTOR_LABELS,
    DETECTOR_MARKERS,
    plot_metric_vs_pilots,
    setup_plot_style,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ICC-style results from CSV tables.")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--target-snr-db", type=float, default=5.0)
    parser.add_argument("--context-pilots", type=int, nargs="*", default=None)
    return parser.parse_args()


def plot_context_metric(context_df: pd.DataFrame, pilot_symbols: int, output_dir: Path, metric: str, target_snr_db: float):
    subset = context_df[context_df["pilot_symbols"] == pilot_symbols].copy()
    if subset.empty:
        return []

    fig, ax = plt.subplots()
    for detector, group in subset.groupby("detector"):
        group = group.sort_values("context_length")
        label = DETECTOR_LABELS.get(detector, detector)
        color = DETECTOR_COLORS.get(detector)
        marker = DETECTOR_MARKERS.get(detector, "o")
        if detector in {"mmse", "icl"}:
            value = group[metric].mean()
            ax.axhline(value, linestyle="--", linewidth=2.0, color=color, label=f"{label} (fixed)")
        else:
            ax.plot(
                group["context_length"],
                group[metric],
                marker=marker,
                linewidth=2.0,
                color=color,
                label=label,
            )

    ax.set_yscale("log")
    ax.set_xlabel("Context sequence length")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} vs. context length, {pilot_symbols} pilots, {target_snr_db:g} dB")
    ax.legend(ncol=2)
    fig.tight_layout()
    png = output_dir / f"{metric}_vs_context_k{pilot_symbols:02d}.png"
    pdf = output_dir / f"{metric}_vs_context_k{pilot_symbols:02d}.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    pilot_df = pd.read_csv(results_dir / "aggregate_by_pilot.csv")
    context_df = pd.read_csv(results_dir / "aggregate_by_context.csv")
    setup_plot_style()

    figures = []
    for metric in ["ser", "ber"]:
        figures.extend(plot_metric_vs_pilots(pilot_df, results_dir, metric, args.target_snr_db))

    context_pilots = args.context_pilots
    if context_pilots is None:
        context_pilots = sorted(int(k) for k in context_df["pilot_symbols"].unique())
    for k in context_pilots:
        for metric in ["ser", "ber"]:
            figures.extend(plot_context_metric(context_df, k, results_dir, metric, args.target_snr_db))

    print("Generated figures:")
    for path in figures:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
