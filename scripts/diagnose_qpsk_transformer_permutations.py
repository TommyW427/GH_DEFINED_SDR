#!/usr/bin/env python3
"""
Diagnostic-only QPSK class-order permutation test for Paper ICL/DEFINED.

This is not an experimental detector for final results. It is intended to
answer one question: do the QPSK checkpoints use a different class-index to
constellation-point mapping than the SDR harness?
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import sys
from pathlib import Path


# The local torch build can crash if NumPy is imported first.
import torch  # noqa: F401

import numpy as np

from core.frame_processing import (
    build_payload_cases_from_phase_result,
    make_detector,
    run_detector_on_cases,
)
from core.modulation import bits_per_symbol, bits_to_symbols, constellation_points
from qpsk_validate_from_post_costas import (
    evaluate_phase,
    load_expected_payloads,
    load_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QPSK Transformer class-order diagnostics.")
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--post-costas-file", default="received_processed_qpsk.bin")
    parser.add_argument("--tx-frame", default="transmitted_frame.bin")
    parser.add_argument("--metadata", default="frame_metadata.json")
    parser.add_argument("--validation-json", default="mqam_validation.json")
    parser.add_argument("--validation-npz", default="mqam_validation.npz")
    parser.add_argument(
        "--rerun-alignment",
        action="store_true",
        help="Ignore saved validation artifacts and rerun phase/alignment search.",
    )
    parser.add_argument("--search-start", type=int, default=200000)
    parser.add_argument("--coarse-step", type=int, default=128)
    parser.add_argument("--refine-radius", type=int, default=64)
    parser.add_argument("--payload-search-radius", type=int, default=12)
    parser.add_argument("--pilots-to-check", type=int, default=1)
    parser.add_argument("--phase-start", type=int, default=0)
    parser.add_argument("--phase-stop", type=int, default=25)
    parser.add_argument("--sps", type=int, default=25)
    parser.add_argument("--output", default="qpsk_transformer_permutation_diagnostic.json")
    return parser.parse_args()


def symbol_error_counts(expected_bits: np.ndarray, detected_bits: np.ndarray, modulation: str) -> tuple[int, int]:
    bits_per = bits_per_symbol(modulation)
    usable = min(len(expected_bits), len(detected_bits))
    usable -= usable % bits_per
    if usable <= 0:
        return 0, 0
    expected_groups = expected_bits[:usable].reshape(-1, bits_per)
    detected_groups = detected_bits[:usable].reshape(-1, bits_per)
    return int(np.sum(np.any(expected_groups != detected_groups, axis=1))), int(len(expected_groups))


def detector_summary(detector, payload_cases) -> dict:
    summary = run_detector_on_cases(detector, payload_cases)
    total_symbol_errors = 0
    total_symbols = 0
    for item, case in zip(summary.payload_results, payload_cases):
        detected_bits = item.run_result.detected_bits[: len(case.expected_bits)]
        sym_errors, symbols = symbol_error_counts(case.expected_bits, detected_bits, case.frame_inputs.config.modulation)
        total_symbol_errors += sym_errors
        total_symbols += symbols
    return {
        "ber": summary.overall_ber,
        "bit_errors": summary.total_errors,
        "bits": summary.total_bits,
        "ser": (total_symbol_errors / total_symbols) if total_symbols else 1.0,
        "symbol_errors": total_symbol_errors,
        "symbols": total_symbols,
    }


def load_callable(module_name: str, callable_name: str):
    module = importlib.import_module(module_name)
    return getattr(module, callable_name)


def select_phase(args: argparse.Namespace, metadata: dict, tx_bits: np.ndarray, recovered: np.ndarray, capture_dir: Path):
    modulation = metadata.get("modulation", "QPSK")
    preamble_bits = tx_bits[: metadata["preamble_bits"]]
    pilot_bits = tx_bits[metadata["preamble_bits"] : metadata["preamble_bits"] + metadata["pilot_bits"]]
    preamble_ref = bits_to_symbols(preamble_bits, modulation)
    pilot_ref = bits_to_symbols(pilot_bits, modulation)
    expected_payload_bits = load_expected_payloads(metadata, capture_dir, tx_bits)

    best = None
    for phase in range(args.phase_start, min(args.phase_stop, args.sps)):
        result = evaluate_phase(
            phase=phase,
            post_costas=recovered,
            sps=args.sps,
            metadata=metadata,
            preamble_ref=preamble_ref,
            pilot_bits=pilot_bits,
            pilot_ref=pilot_ref,
            expected_payload_bits=expected_payload_bits,
            search_start=args.search_start,
            coarse_step=args.coarse_step,
            refine_radius=args.refine_radius,
            payload_search_radius=args.payload_search_radius,
            pilots_to_check=args.pilots_to_check,
        )
        sort_key = (-result["lock_score"], result["overall_ber"], result["mean_pilot_ber"])
        if best is None or sort_key < best["sort_key"]:
            best = {"sort_key": sort_key, "result": result}
    return best["result"], expected_payload_bits


def load_saved_phase_result(args: argparse.Namespace, capture_dir: Path) -> dict | None:
    validation_path = capture_dir / args.validation_json
    symbols_path = capture_dir / args.validation_npz
    if args.rerun_alignment or not validation_path.exists() or not symbols_path.exists():
        return None
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    best_phase = validation.get("best_phase")
    if not best_phase:
        return None
    symbols = np.load(symbols_path, allow_pickle=True)
    result = dict(best_phase)
    result["phase"] = result.get("phase", validation.get("phase"))
    result["decimated_symbols"] = symbols["decimated_symbols"]
    return result


def main() -> int:
    args = parse_args()
    capture_dir = Path(args.capture_dir)
    metadata = load_metadata(capture_dir / args.metadata)
    if metadata.get("modulation") != "QPSK":
        raise ValueError("This diagnostic is QPSK-only.")

    tx_bits = np.fromfile(capture_dir / args.tx_frame, dtype=np.uint8)
    expected_payload_bits = load_expected_payloads(metadata, capture_dir, tx_bits)
    phase_result = load_saved_phase_result(args, capture_dir)
    if phase_result is None:
        recovered = np.fromfile(capture_dir / args.post_costas_file, dtype=np.complex64)
        phase_result, expected_payload_bits = select_phase(args, metadata, tx_bits, recovered, capture_dir)
    payload_cases = build_payload_cases_from_phase_result(
        phase_result=phase_result,
        decimated_symbols=phase_result["decimated_symbols"],
        metadata=metadata,
        tx_bits=tx_bits,
        expected_payload_bits=expected_payload_bits,
    )
    if not payload_cases:
        raise RuntimeError("No payload cases could be constructed.")

    config = payload_cases[0].frame_inputs.config
    base_points = np.array(constellation_points("QPSK"), dtype=np.complex64)

    import paper_transformer_backend

    icl_runtime = paper_transformer_backend.get_runtime("icl", "QPSK", total_symbols=31)
    defined_runtime = paper_transformer_backend.get_runtime("defined", "QPSK", total_symbols=31)
    icl_callable = load_callable("PaperICL", "detect_symbol_with_context")
    defined_callable = load_callable("PaperDEFINED", "detect_symbol_with_context")

    baselines = {
        "mmse": detector_summary(make_detector("mmse", config), payload_cases),
        "mmse_df": detector_summary(make_detector("mmse_df", config), payload_cases),
    }

    rows = []
    for perm in itertools.permutations(range(4)):
        permuted = base_points[list(perm)]
        icl_runtime.constellation = permuted
        defined_runtime.constellation = permuted
        icl_summary = detector_summary(
            make_detector("icl", config, icl_model_func=icl_callable),
            payload_cases,
        )
        defined_summary = detector_summary(
            make_detector("defined", config, defined_model_func=defined_callable),
            payload_cases,
        )
        rows.append(
            {
                "permutation": list(perm),
                "class_to_symbol": [str(complex(point)) for point in permuted],
                "icl": icl_summary,
                "defined": defined_summary,
            }
        )

    rows.sort(key=lambda row: (row["defined"]["ber"], row["icl"]["ber"], row["defined"]["ser"], row["icl"]["ser"]))
    output = {
        "capture_dir": str(capture_dir),
        "pilot_symbols": metadata["pilot_symbols"],
        "phase_selection": {
            "best_phase": phase_result["phase"],
            "lock_score": phase_result["lock_score"],
            "mean_pilot_ber": phase_result["mean_pilot_ber"],
            "mean_snr_db": phase_result.get("mean_snr_db"),
            "overall_ber": phase_result["overall_ber"],
        },
        "baseline": baselines,
        "permutation_results": rows,
    }

    output_path = capture_dir / args.output
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("=" * 72)
    print("QPSK TRANSFORMER PERMUTATION DIAGNOSTIC")
    print("=" * 72)
    print(f"Capture dir:   {capture_dir}")
    print(f"Pilot symbols: {metadata['pilot_symbols']}")
    print(f"Lock score:    {phase_result['lock_score']:.4f}")
    if phase_result.get("mean_snr_db") is not None:
        print(f"Mean SNR:      {phase_result['mean_snr_db']:.2f} dB")
    print()
    print(f"mmse:    BER={baselines['mmse']['ber']:.6f}, SER={baselines['mmse']['ser']:.6f}")
    print(f"mmse_df: BER={baselines['mmse_df']['ber']:.6f}, SER={baselines['mmse_df']['ser']:.6f}")
    print()
    print("Top permutations by DEFINED then ICL BER:")
    for row in rows[:8]:
        print(
            f"  perm={row['permutation']} "
            f"iclBER={row['icl']['ber']:.6f} iclSER={row['icl']['ser']:.6f} "
            f"definedBER={row['defined']['ber']:.6f} definedSER={row['defined']['ser']:.6f}"
        )
    print()
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
