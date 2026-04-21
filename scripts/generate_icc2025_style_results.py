#!/usr/bin/env python3
"""
Generate ICC-2025-style SDR/DEFINED result tables and plots.

The ICC paper's main visual style is SER versus context sequence length for
fixed SNR/pilot settings, with MMSE baselines and decision-feedback curves.
This script produces analogous artifacts from already validated SDR captures,
optionally adding offline AWGN to create a controlled SNR condition.
"""

from __future__ import annotations

import argparse
import glob
import importlib
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path


def _argv_requests_torch_backend(argv: list[str]) -> bool:
    if "--detectors" not in argv:
        return True
    start = argv.index("--detectors") + 1
    detectors = []
    for item in argv[start:]:
        if item.startswith("--"):
            break
        detectors.append(item.lower())
    return "icl" in detectors or "defined" in detectors


if _argv_requests_torch_backend(sys.argv):
    # This environment's torch build can crash if NumPy is imported first.
    import torch  # noqa: F401

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.frame_processing import build_payload_cases_from_phase_result, make_detector
from core.modulation import bits_per_symbol
from qpsk_validate_from_post_costas import load_expected_payloads, load_metadata
from run_offline_awgn_experiment import (
    add_awgn_to_cases,
    detector_summary,
    load_saved_phase_result,
    symbol_error_counts,
)


DETECTOR_LABELS = {
    "mmse": "MMSE",
    "mmse_df": "MMSE-DF",
    "icl": "DEFINED-ICL",
    "defined": "DEFINED-DF",
}

DETECTOR_COLORS = {
    "mmse": "#1f2937",
    "mmse_df": "#2563eb",
    "icl": "#f97316",
    "defined": "#16a34a",
}

DETECTOR_MARKERS = {
    "mmse": "s",
    "mmse_df": "o",
    "icl": "^",
    "defined": "D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ICC-2025-style data tables and figures.")
    parser.add_argument("--captures-root", default="captures")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--modulation", default="QPSK", choices=["BPSK", "QPSK", "16QAM", "64QAM"])
    parser.add_argument("--pilot-symbols", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    parser.add_argument("--context-pilot", type=int, default=2, help="Pilot count for SER-vs-context plot.")
    parser.add_argument("--target-snr-db", type=float, default=5.0)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--max-captures-per-k", type=int, default=8)
    parser.add_argument("--lock-threshold", type=float, default=0.95)
    parser.add_argument("--max-clean-validation-ber", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--detectors", nargs="+", default=["mmse", "mmse_df", "icl", "defined"])
    parser.add_argument("--validation-json", default="mqam_validation.json")
    parser.add_argument("--validation-npz", default="mqam_validation.npz")
    parser.add_argument("--tx-frame", default="transmitted_frame.bin")
    parser.add_argument("--metadata", default="frame_metadata.json")
    return parser.parse_args()


def load_callable(module_name: str, callable_name: str):
    module = importlib.import_module(module_name)
    return getattr(module, callable_name)


def pilot_from_dir(path: Path) -> int | None:
    match = re.search(r"_k(\d{2})", path.name)
    return int(match.group(1)) if match else None


def candidate_capture_dirs(root: Path, modulation: str, pilot_symbols: list[int]) -> list[Path]:
    dirs: set[Path] = set()
    mod_tag = modulation.lower()
    for k in pilot_symbols:
        patterns = [
            str(root / f"*{mod_tag}_paper_k{k:02d}*" / "mqam_validation.json"),
            str(root / f"*{mod_tag}_paper_k{k:02d}*" / "qpsk_validation.json"),
        ]
        for pattern in patterns:
            for validation_path in glob.glob(pattern):
                dirs.add(Path(validation_path).parent)
    return sorted(dirs)


def load_capture(capture_dir: Path, args: argparse.Namespace):
    metadata_path = capture_dir / args.metadata
    tx_path = capture_dir / args.tx_frame
    if not metadata_path.exists() or not tx_path.exists():
        return None
    metadata = load_metadata(metadata_path)
    if metadata.get("modulation") != args.modulation:
        return None
    tx_bits = np.fromfile(tx_path, dtype=np.uint8)
    try:
        phase_result = load_saved_phase_result(capture_dir, args.validation_json, args.validation_npz)
    except Exception:
        return None
    if float(phase_result.get("lock_score", 0.0)) < args.lock_threshold:
        return None
    if float(phase_result.get("overall_ber", 1.0)) > args.max_clean_validation_ber:
        return None
    expected_payload_bits = load_expected_payloads(metadata, capture_dir, tx_bits)
    payload_cases = build_payload_cases_from_phase_result(
        phase_result=phase_result,
        decimated_symbols=phase_result["decimated_symbols"],
        metadata=metadata,
        tx_bits=tx_bits,
        expected_payload_bits=expected_payload_bits,
    )
    if not payload_cases:
        return None
    return {
        "capture_dir": str(capture_dir),
        "metadata": metadata,
        "phase_result": phase_result,
        "payload_cases": payload_cases,
    }


def select_captures(args: argparse.Namespace) -> list[dict]:
    root = Path(args.captures_root)
    selected = []
    counts = defaultdict(int)
    for capture_dir in candidate_capture_dirs(root, args.modulation, args.pilot_symbols):
        loaded = load_capture(capture_dir, args)
        if loaded is None:
            continue
        k = int(loaded["metadata"]["pilot_symbols"])
        if counts[k] >= args.max_captures_per_k:
            continue
        selected.append(loaded)
        counts[k] += 1
    return selected


def run_detector(detector_name: str, config, icl_callable, defined_callable):
    return make_detector(
        detector_name,
        config,
        icl_model_func=icl_callable,
        defined_model_func=defined_callable,
    )


def step_rows_for_run(detector_name: str, run_result, expected_bits: np.ndarray, metadata: dict, repetition: int, capture_dir: str):
    bps = bits_per_symbol(metadata["modulation"])
    rows = []
    for step in run_result.steps:
        start = step.index * bps
        stop = start + bps
        expected_symbol_bits = expected_bits[start:stop]
        detected_symbol_bits = step.detected_bits[: len(expected_symbol_bits)]
        bit_errors = int(np.sum(detected_symbol_bits != expected_symbol_bits))
        rows.append(
            {
                "capture_dir": capture_dir,
                "repetition": repetition,
                "pilot_symbols": int(metadata["pilot_symbols"]),
                "data_symbol_index": int(step.index),
                "context_length": int(step.context_length),
                "detector": detector_name,
                "bit_errors": bit_errors,
                "bits": int(len(expected_symbol_bits)),
                "symbol_error": int(bit_errors > 0),
            }
        )
    return rows


def aggregate(rows: list[dict], group_cols: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if "symbol_error" in df.columns:
        grouped = df.groupby(group_cols, dropna=False).agg(
            bit_errors=("bit_errors", "sum"),
            bits=("bits", "sum"),
            symbol_errors=("symbol_error", "sum"),
            symbols=("symbol_error", "count"),
        )
    else:
        grouped = df.groupby(group_cols, dropna=False).agg(
            bit_errors=("bit_errors", "sum"),
            bits=("bits", "sum"),
            symbol_errors=("symbol_errors", "sum"),
            symbols=("symbols", "sum"),
        )
    grouped = grouped.reset_index()
    grouped["ber"] = grouped["bit_errors"] / grouped["bits"].clip(lower=1)
    grouped["ser"] = grouped["symbol_errors"] / grouped["symbols"].clip(lower=1)
    return grouped


def run_experiment(args: argparse.Namespace, captures: list[dict]):
    detector_names = [name.lower() for name in args.detectors]
    icl_callable = load_callable("PaperICL", "detect_symbol_with_context") if "icl" in detector_names else None
    defined_callable = load_callable("PaperDEFINED", "detect_symbol_with_context") if "defined" in detector_names else None
    rng = np.random.default_rng(args.seed)

    clean_rows = []
    noisy_summary_rows = []
    noisy_step_rows = []
    snr_rows = []

    for capture in captures:
        metadata = capture["metadata"]
        payload_cases = capture["payload_cases"]
        config = payload_cases[0].frame_inputs.config
        capture_dir = capture["capture_dir"]
        clean_record = {
            "capture_dir": capture_dir,
            "pilot_symbols": int(metadata["pilot_symbols"]),
            "lock_score": float(capture["phase_result"]["lock_score"]),
            "clean_validation_ber": float(capture["phase_result"].get("overall_ber", math.nan)),
            "clean_validation_mean_snr_db": capture["phase_result"].get("mean_snr_db"),
        }
        for detector_name in detector_names:
            detector = run_detector(detector_name, config, icl_callable, defined_callable)
            summary = detector_summary(detector, payload_cases)
            clean_record[f"{detector_name}_ber"] = summary["ber"]
            clean_record[f"{detector_name}_ser"] = summary["ser"]
        clean_rows.append(clean_record)

        for rep in range(args.repetitions):
            noisy_cases, noise_rows = add_awgn_to_cases(payload_cases, args.target_snr_db, rng)
            snr_rows.extend(
                {
                    "capture_dir": capture_dir,
                    "repetition": rep,
                    "pilot_symbols": int(metadata["pilot_symbols"]),
                    **row,
                }
                for row in noise_rows
            )
            for detector_name in detector_names:
                detector = run_detector(detector_name, config, icl_callable, defined_callable)
                summary = detector_summary(detector, noisy_cases)
                noisy_summary_rows.append(
                    {
                        "capture_dir": capture_dir,
                        "repetition": rep,
                        "pilot_symbols": int(metadata["pilot_symbols"]),
                        "target_snr_db": args.target_snr_db,
                        "detector": detector_name,
                        **summary,
                    }
                )
                for case in noisy_cases:
                    run_result = detector.detect(case.frame_inputs)
                    noisy_step_rows.extend(
                        step_rows_for_run(
                            detector_name=detector_name,
                            run_result=run_result,
                            expected_bits=case.expected_bits,
                            metadata=metadata,
                            repetition=rep,
                            capture_dir=capture_dir,
                        )
                    )

    return {
        "clean": pd.DataFrame(clean_rows),
        "noisy_summary": pd.DataFrame(noisy_summary_rows),
        "noisy_steps": pd.DataFrame(noisy_step_rows),
        "snr": pd.DataFrame(snr_rows),
    }


def save_tables(output_dir: Path, data: dict[str, pd.DataFrame]) -> dict[str, Path]:
    paths = {}
    for name, df in data.items():
        path = output_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        paths[name] = path
    pilot_agg = aggregate(data["noisy_summary"].to_dict("records"), ["pilot_symbols", "detector", "target_snr_db"])
    context_agg = aggregate(data["noisy_steps"].to_dict("records"), ["pilot_symbols", "context_length", "detector"])
    pilot_agg_path = output_dir / "aggregate_by_pilot.csv"
    context_agg_path = output_dir / "aggregate_by_context.csv"
    pilot_agg.to_csv(pilot_agg_path, index=False)
    context_agg.to_csv(context_agg_path, index=False)
    paths["aggregate_by_pilot"] = pilot_agg_path
    paths["aggregate_by_context"] = context_agg_path
    return paths


def setup_plot_style():
    plt.rcParams.update(
        {
            "figure.figsize": (7.2, 4.6),
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
        }
    )


def plot_metric_vs_pilots(df: pd.DataFrame, output_dir: Path, metric: str, target_snr_db: float):
    fig, ax = plt.subplots()
    for detector, group in df.groupby("detector"):
        group = group.sort_values("pilot_symbols")
        ax.plot(
            group["pilot_symbols"],
            group[metric],
            marker=DETECTOR_MARKERS.get(detector, "o"),
            linewidth=2.0,
            color=DETECTOR_COLORS.get(detector),
            label=DETECTOR_LABELS.get(detector, detector),
        )
    ax.set_yscale("log")
    ax.set_xlabel("Number of pilot symbols")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} vs. pilots, {target_snr_db:g} dB offline AWGN")
    ax.set_xticks(sorted(df["pilot_symbols"].unique()))
    ax.legend(ncol=2)
    fig.tight_layout()
    png = output_dir / f"{metric}_vs_pilots.png"
    pdf = output_dir / f"{metric}_vs_pilots.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def plot_context_ser(context_df: pd.DataFrame, pilot_symbols: int, output_dir: Path, target_snr_db: float):
    subset = context_df[context_df["pilot_symbols"] == pilot_symbols].copy()
    if subset.empty:
        return None
    fig, ax = plt.subplots()

    for detector, group in subset.groupby("detector"):
        group = group.sort_values("context_length")
        label = DETECTOR_LABELS.get(detector, detector)
        color = DETECTOR_COLORS.get(detector)
        marker = DETECTOR_MARKERS.get(detector, "o")
        if detector in {"mmse", "icl"}:
            value = group["ser"].mean()
            ax.axhline(value, linestyle="--", linewidth=2.0, color=color, label=f"{label} (fixed)")
        else:
            ax.plot(
                group["context_length"],
                group["ser"],
                marker=marker,
                linewidth=2.0,
                color=color,
                label=label,
            )

    ax.set_yscale("log")
    ax.set_xlabel("Context sequence length")
    ax.set_ylabel("SER")
    ax.set_title(f"SER vs. context length, {pilot_symbols} pilots, {target_snr_db:g} dB")
    ax.legend(ncol=2)
    fig.tight_layout()
    png = output_dir / f"ser_vs_context_k{pilot_symbols:02d}.png"
    pdf = output_dir / f"ser_vs_context_k{pilot_symbols:02d}.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def write_summary(output_dir: Path, args: argparse.Namespace, captures: list[dict], tables: dict[str, Path], figures: list[Path]):
    summary = {
        "modulation": args.modulation,
        "target_snr_db": args.target_snr_db,
        "repetitions": args.repetitions,
        "pilot_symbols": args.pilot_symbols,
        "context_pilot": args.context_pilot,
        "selected_captures": [
            {
                "capture_dir": item["capture_dir"],
                "pilot_symbols": item["metadata"]["pilot_symbols"],
                "lock_score": item["phase_result"].get("lock_score"),
                "clean_validation_ber": item["phase_result"].get("overall_ber"),
                "clean_validation_mean_snr_db": item["phase_result"].get("mean_snr_db"),
            }
            for item in captures
        ],
        "tables": {name: str(path) for name, path in tables.items()},
        "figures": [str(path) for path in figures],
        "notes": [
            "Offline AWGN is added after synchronization to aligned SDR symbols.",
            "MMSE and ICL curves in context plots are fixed-context horizontal baselines.",
            "MMSE-DF and DEFINED-DF curves use detected symbols as decision-feedback context.",
        ],
    }
    (output_dir / "paper_style_results_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_lines = [
        "# ICC 2025 Style SDR Results",
        "",
        f"- Modulation: `{args.modulation}`",
        f"- Target offline AWGN SNR: `{args.target_snr_db:g} dB`",
        f"- Repetitions per capture: `{args.repetitions}`",
        f"- Selected captures: `{len(captures)}`",
        "",
        "## Tables",
        "",
    ]
    for name, path in tables.items():
        md_lines.append(f"- `{name}`: `{path.name}`")
    md_lines.extend(["", "## Figures", ""])
    for path in figures:
        md_lines.append(f"- `{path.name}`")
    md_lines.extend(
        [
            "",
            "## Method Note",
            "",
            "These plots intentionally isolate detector behavior by adding AWGN after SDR synchronization/alignment. "
            "They should be reported separately from fully empirical over-the-air SNR sweeps.",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(md_lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir or f"paper_results/icc2025_style_{time.strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=False)

    captures = select_captures(args)
    if not captures:
        raise RuntimeError("No validated captures matched the requested filters.")

    data = run_experiment(args, captures)
    tables = save_tables(output_dir, data)

    setup_plot_style()
    pilot_agg = pd.read_csv(tables["aggregate_by_pilot"])
    context_agg = pd.read_csv(tables["aggregate_by_context"])
    figures = []
    for metric in ["ser", "ber"]:
        figures.extend(plot_metric_vs_pilots(pilot_agg, output_dir, metric, args.target_snr_db))
    context_figs = plot_context_ser(context_agg, args.context_pilot, output_dir, args.target_snr_db)
    if context_figs:
        figures.extend(context_figs)

    write_summary(output_dir, args, captures, tables, figures)

    print("=" * 72)
    print("ICC 2025 STYLE RESULT GENERATION")
    print("=" * 72)
    print(f"Output dir:        {output_dir}")
    print(f"Selected captures: {len(captures)}")
    by_k = defaultdict(int)
    for item in captures:
        by_k[int(item["metadata"]["pilot_symbols"])] += 1
    for k in sorted(by_k):
        print(f"  k={k}: {by_k[k]} captures")
    print()
    print("Tables:")
    for name, path in tables.items():
        print(f"  {name}: {path}")
    print()
    print("Figures:")
    for path in figures:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
