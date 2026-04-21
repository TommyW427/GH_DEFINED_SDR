#!/usr/bin/env python3
"""
Batch QPSK class-order diagnostic across validated k=2/k=4 SDR captures.

This script aggregates the diagnostic-only permutation test from
diagnose_qpsk_transformer_permutations.py over multiple high-lock captures.
It is meant to identify the checkpoint's QPSK class mapping, not to produce
final paper-style BER curves.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
from collections import defaultdict
from pathlib import Path


# Keep torch before NumPy in this environment.
import torch  # noqa: F401

import numpy as np

import paper_transformer_backend
from core.frame_processing import (
    build_payload_cases_from_phase_result,
    make_detector,
)
from core.modulation import constellation_points
from diagnose_qpsk_transformer_permutations import detector_summary, load_callable, load_saved_phase_result
from qpsk_validate_from_post_costas import load_expected_payloads, load_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch QPSK Transformer class-order diagnostic.")
    parser.add_argument("--captures-root", default="captures")
    parser.add_argument("--pilot-symbols", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--lock-threshold", type=float, default=0.95)
    parser.add_argument("--max-mmse-ber", type=float, default=0.0)
    parser.add_argument("--max-captures-per-k", type=int, default=8)
    parser.add_argument("--output", default="qpsk_transformer_permutation_batch.json")
    return parser.parse_args()


def candidate_capture_dirs(root: Path, pilot_symbols: list[int]) -> list[Path]:
    dirs: set[Path] = set()
    for k in pilot_symbols:
        pattern = str(root / f"*qpsk_paper_k{k:02d}_r*" / "mqam_validation.json")
        for path in glob.glob(pattern):
            dirs.add(Path(path).parent)
    return sorted(dirs)


def load_capture_case(capture_dir: Path):
    metadata = load_metadata(capture_dir / "frame_metadata.json")
    if metadata.get("modulation") != "QPSK":
        return None
    tx_bits = np.fromfile(capture_dir / "transmitted_frame.bin", dtype=np.uint8)
    expected_payload_bits = load_expected_payloads(metadata, capture_dir, tx_bits)

    class Args:
        validation_json = "mqam_validation.json"
        validation_npz = "mqam_validation.npz"
        rerun_alignment = False

    phase_result = load_saved_phase_result(Args, capture_dir)
    if phase_result is None:
        return None
    payload_cases = build_payload_cases_from_phase_result(
        phase_result=phase_result,
        decimated_symbols=phase_result["decimated_symbols"],
        metadata=metadata,
        tx_bits=tx_bits,
        expected_payload_bits=expected_payload_bits,
    )
    if not payload_cases:
        return None
    return metadata, phase_result, payload_cases


def main() -> int:
    args = parse_args()
    root = Path(args.captures_root)
    base_points = np.array(constellation_points("QPSK"), dtype=np.complex64)
    permutations = list(itertools.permutations(range(4)))

    icl_runtime = paper_transformer_backend.get_runtime("icl", "QPSK", total_symbols=31)
    defined_runtime = paper_transformer_backend.get_runtime("defined", "QPSK", total_symbols=31)
    icl_callable = load_callable("PaperICL", "detect_symbol_with_context")
    defined_callable = load_callable("PaperDEFINED", "detect_symbol_with_context")

    selected = []
    skipped = []
    per_k_counts = defaultdict(int)
    for capture_dir in candidate_capture_dirs(root, args.pilot_symbols):
        loaded = load_capture_case(capture_dir)
        if loaded is None:
            skipped.append({"capture_dir": str(capture_dir), "reason": "missing_or_invalid_artifacts"})
            continue
        metadata, phase_result, payload_cases = loaded
        k = int(metadata["pilot_symbols"])
        if per_k_counts[k] >= args.max_captures_per_k:
            continue
        config = payload_cases[0].frame_inputs.config
        mmse = detector_summary(make_detector("mmse", config), payload_cases)
        lock_score = float(phase_result.get("lock_score", 0.0))
        if lock_score < args.lock_threshold:
            skipped.append({"capture_dir": str(capture_dir), "reason": "low_lock", "lock_score": lock_score})
            continue
        if mmse["ber"] > args.max_mmse_ber:
            skipped.append({"capture_dir": str(capture_dir), "reason": "mmse_not_clean", "mmse_ber": mmse["ber"]})
            continue
        selected.append((capture_dir, metadata, phase_result, payload_cases, mmse))
        per_k_counts[k] += 1

    aggregate = {
        tuple(perm): {
            "permutation": list(perm),
            "captures": 0,
            "icl_bit_errors": 0,
            "icl_bits": 0,
            "icl_symbol_errors": 0,
            "icl_symbols": 0,
            "defined_bit_errors": 0,
            "defined_bits": 0,
            "defined_symbol_errors": 0,
            "defined_symbols": 0,
            "capture_rows": [],
        }
        for perm in permutations
    }

    for capture_dir, metadata, phase_result, payload_cases, mmse in selected:
        config = payload_cases[0].frame_inputs.config
        for perm in permutations:
            permuted = base_points[list(perm)]
            icl_runtime.constellation = permuted
            defined_runtime.constellation = permuted
            icl = detector_summary(make_detector("icl", config, icl_model_func=icl_callable), payload_cases)
            defined = detector_summary(make_detector("defined", config, defined_model_func=defined_callable), payload_cases)
            row = aggregate[tuple(perm)]
            row["captures"] += 1
            row["icl_bit_errors"] += icl["bit_errors"]
            row["icl_bits"] += icl["bits"]
            row["icl_symbol_errors"] += icl["symbol_errors"]
            row["icl_symbols"] += icl["symbols"]
            row["defined_bit_errors"] += defined["bit_errors"]
            row["defined_bits"] += defined["bits"]
            row["defined_symbol_errors"] += defined["symbol_errors"]
            row["defined_symbols"] += defined["symbols"]
            row["capture_rows"].append(
                {
                    "capture_dir": str(capture_dir),
                    "pilot_symbols": int(metadata["pilot_symbols"]),
                    "lock_score": float(phase_result["lock_score"]),
                    "mean_snr_db": phase_result.get("mean_snr_db"),
                    "mmse_ber": mmse["ber"],
                    "icl_ber": icl["ber"],
                    "icl_ser": icl["ser"],
                    "defined_ber": defined["ber"],
                    "defined_ser": defined["ser"],
                }
            )

    results = []
    for row in aggregate.values():
        if row["captures"] == 0:
            continue
        row["icl_ber"] = row["icl_bit_errors"] / row["icl_bits"] if row["icl_bits"] else 1.0
        row["icl_ser"] = row["icl_symbol_errors"] / row["icl_symbols"] if row["icl_symbols"] else 1.0
        row["defined_ber"] = row["defined_bit_errors"] / row["defined_bits"] if row["defined_bits"] else 1.0
        row["defined_ser"] = row["defined_symbol_errors"] / row["defined_symbols"] if row["defined_symbols"] else 1.0
        results.append(row)

    results.sort(key=lambda row: (row["defined_ber"], row["icl_ber"], row["defined_ser"], row["icl_ser"]))

    output = {
        "filters": {
            "pilot_symbols": args.pilot_symbols,
            "lock_threshold": args.lock_threshold,
            "max_mmse_ber": args.max_mmse_ber,
            "max_captures_per_k": args.max_captures_per_k,
        },
        "selected_captures": [
            {
                "capture_dir": str(capture_dir),
                "pilot_symbols": int(metadata["pilot_symbols"]),
                "lock_score": float(phase_result["lock_score"]),
                "mean_snr_db": phase_result.get("mean_snr_db"),
                "mmse_ber": mmse["ber"],
            }
            for capture_dir, metadata, phase_result, payload_cases, mmse in selected
        ],
        "skipped_captures": skipped,
        "permutation_results": results,
    }
    output_path = Path(args.output)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("=" * 72)
    print("BATCH QPSK TRANSFORMER PERMUTATION DIAGNOSTIC")
    print("=" * 72)
    print(f"Selected captures: {len(selected)}")
    for k in sorted(per_k_counts):
        print(f"  k={k}: {per_k_counts[k]}")
    print()
    print("Top aggregate permutations by DEFINED then ICL BER:")
    for row in results[:10]:
        print(
            f"  perm={row['permutation']} captures={row['captures']} "
            f"iclBER={row['icl_ber']:.6f} iclSER={row['icl_ser']:.6f} "
            f"definedBER={row['defined_ber']:.6f} definedSER={row['defined_ser']:.6f}"
        )
    print()
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
