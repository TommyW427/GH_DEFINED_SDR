#!/usr/bin/env python3
"""
Run modular detector experiments on an aligned SDR capture.

This script:
  1. uses the post-Costas validator to find the best fixed sample phase
  2. builds per-payload detector inputs
  3. runs one or more detector backends through the unified interface
  4. writes a compact JSON summary
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path


def _argv_requests_torch_backend(argv: list[str]) -> bool:
    if "--detectors" not in argv:
        return False
    start = argv.index("--detectors") + 1
    detectors = []
    for item in argv[start:]:
        if item.startswith("--"):
            break
        detectors.append(item.lower())
    return "icl" in detectors or "defined" in detectors


if _argv_requests_torch_backend(sys.argv):
    # This environment's torch build can crash if NumPy is imported first.
    # Import torch before any project module that imports NumPy.
    import torch  # noqa: F401

import numpy as np

from core.frame_processing import (
    make_detector,
    build_payload_cases_from_phase_result,
    run_detector_on_cases,
)
from core.modulation import bits_per_symbol, bits_to_symbols
from qpsk_validate_from_post_costas import (
    load_metadata,
    load_expected_payloads,
    evaluate_phase,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run detector backends on an aligned SDR capture.")
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--post-costas-file", default="received_post_costas_qpsk.bin")
    parser.add_argument("--tx-frame", default="transmitted_frame.bin")
    parser.add_argument("--metadata", default="frame_metadata.json")
    parser.add_argument("--validation-json", default="mqam_validation.json")
    parser.add_argument("--validation-npz", default="mqam_validation.npz")
    parser.add_argument(
        "--rerun-alignment",
        action="store_true",
        help="Ignore saved validation artifacts and rerun phase/alignment search.",
    )
    parser.add_argument("--detectors", nargs="+", default=["mmse", "mmse_df"])
    parser.add_argument("--icl-module", default=None, help="Optional module path for ICL callable")
    parser.add_argument("--icl-callable", default=None, help="Callable name inside --icl-module")
    parser.add_argument("--icl-checkpoint", default=None, help="Checkpoint path for PaperICL backend")
    parser.add_argument("--defined-module", default=None, help="Optional module path for DEFINED callable")
    parser.add_argument("--defined-callable", default=None, help="Callable name inside --defined-module")
    parser.add_argument("--defined-checkpoint", default=None, help="Checkpoint path for PaperDEFINED backend")
    parser.add_argument("--search-start", type=int, default=200000)
    parser.add_argument("--coarse-step", type=int, default=128)
    parser.add_argument("--refine-radius", type=int, default=64)
    parser.add_argument("--payload-search-radius", type=int, default=12)
    parser.add_argument("--pilots-to-check", type=int, default=4)
    parser.add_argument("--phase-start", type=int, default=0)
    parser.add_argument("--phase-stop", type=int, default=25)
    parser.add_argument("--sps", type=int, default=25)
    parser.add_argument("--output", default="receiver_experiment.json")
    return parser.parse_args()


def load_defined_callable(module_name: str | None, callable_name: str | None):
    if not module_name or not callable_name:
        return None
    module = importlib.import_module(module_name)
    return getattr(module, callable_name)


def symbol_error_counts(expected_bits: np.ndarray, detected_bits: np.ndarray, modulation: str) -> tuple[int, int]:
    k = bits_per_symbol(modulation)
    usable = min(len(expected_bits), len(detected_bits))
    usable -= usable % k
    if usable <= 0:
        return 0, 0
    expected_groups = expected_bits[:usable].reshape(-1, k)
    detected_groups = detected_bits[:usable].reshape(-1, k)
    symbol_errors = int(np.sum(np.any(expected_groups != detected_groups, axis=1)))
    return symbol_errors, int(len(expected_groups))


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
    metadata_path = capture_dir / args.metadata
    tx_frame_path = capture_dir / args.tx_frame
    post_costas_path = capture_dir / args.post_costas_file

    metadata = load_metadata(metadata_path)
    tx_bits = np.fromfile(tx_frame_path, dtype=np.uint8)
    expected_payload_bits = load_expected_payloads(metadata, capture_dir, tx_bits)

    phase_result = load_saved_phase_result(args, capture_dir)
    if phase_result is None:
        post_costas = np.fromfile(post_costas_path, dtype=np.complex64)
        preamble_bits = tx_bits[:metadata["preamble_bits"]]
        pilot_bits = tx_bits[metadata["preamble_bits"]:metadata["preamble_bits"] + metadata["pilot_bits"]]
        preamble_ref = bits_to_symbols(preamble_bits, metadata.get("modulation", "QPSK"))
        pilot_ref = bits_to_symbols(pilot_bits, metadata.get("modulation", "QPSK"))

        best = None
        for phase in range(args.phase_start, min(args.phase_stop, args.sps)):
            result = evaluate_phase(
                phase=phase,
                post_costas=post_costas,
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

        phase_result = best["result"]
    payload_cases = build_payload_cases_from_phase_result(
        phase_result=phase_result,
        decimated_symbols=phase_result["decimated_symbols"],
        metadata=metadata,
        tx_bits=tx_bits,
        expected_payload_bits=expected_payload_bits,
    )
    if not payload_cases:
        raise RuntimeError("No payload cases could be constructed from the aligned capture.")

    detector_names = [name.lower() for name in args.detectors]
    if args.icl_checkpoint:
        os.environ["PAPER_ICL_CHECKPOINT"] = args.icl_checkpoint
    if args.defined_checkpoint:
        os.environ["PAPER_DEFINED_CHECKPOINT"] = args.defined_checkpoint

    icl_module = args.icl_module
    icl_callable_name = args.icl_callable
    if "icl" in detector_names and not icl_module:
        icl_module = "PaperICL"
        icl_callable_name = "detect_symbol_with_context"
    defined_module = args.defined_module
    defined_callable_name = args.defined_callable
    if "defined" in detector_names and not defined_module:
        defined_module = "PaperDEFINED"
        defined_callable_name = "detect_symbol_with_context"

    icl_callable = load_defined_callable(icl_module, icl_callable_name)
    defined_callable = load_defined_callable(defined_module, defined_callable_name)
    config = payload_cases[0].frame_inputs.config

    summaries = {}
    for detector_name in args.detectors:
        detector = make_detector(
            detector_name,
            config,
            defined_model_func=defined_callable,
            icl_model_func=icl_callable,
        )
        summary = run_detector_on_cases(detector, payload_cases)
        payload_rows = []
        total_symbol_errors = 0
        total_symbols = 0
        for item, case in zip(summary.payload_results, payload_cases):
            detected_bits = item.run_result.detected_bits[: len(case.expected_bits)]
            symbol_errors, symbols = symbol_error_counts(case.expected_bits, detected_bits, config.modulation)
            total_symbol_errors += symbol_errors
            total_symbols += symbols
            payload_rows.append(
                {
                    "payload_index": item.payload_index,
                    "expected_payload_index": item.expected_payload_index,
                    "ber": item.ber,
                    "bit_errors": item.bit_errors,
                    "bits": item.bits,
                    "ser": (symbol_errors / symbols) if symbols else 1.0,
                    "symbol_errors": symbol_errors,
                    "symbols": symbols,
                    "pilot_score": item.pilot_score,
                    "pilot_ber": item.pilot_ber,
                    "pilot_drift": item.pilot_drift,
                }
            )
        summaries[detector_name] = {
            "overall_ber": summary.overall_ber,
            "total_errors": summary.total_errors,
            "total_bits": summary.total_bits,
            "overall_ser": (total_symbol_errors / total_symbols) if total_symbols else 1.0,
            "total_symbol_errors": total_symbol_errors,
            "total_symbols": total_symbols,
            "payload_results": payload_rows,
        }

    output = {
        "capture_dir": str(capture_dir),
        "post_costas_file": str(post_costas_path),
        "phase_selection": {
            "best_phase": phase_result["phase"],
            "frame_start": phase_result["frame_start"],
            "pilot_start": phase_result["pilot_start"],
            "lock_score": phase_result["lock_score"],
            "payload_cycle_offset": phase_result["payload_cycle_offset"],
            "mean_pilot_ber": phase_result["mean_pilot_ber"],
            "overall_ber": phase_result["overall_ber"],
        },
        "detectors": summaries,
    }

    output_path = capture_dir / args.output
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("=" * 72)
    print("RECEIVER EXPERIMENT")
    print("=" * 72)
    print(f"Capture dir:   {capture_dir}")
    print(f"Best phase:    {phase_result['phase']}")
    print(f"Lock score:    {phase_result['lock_score']:.4f}")
    print(f"Phase BER:     {phase_result['overall_ber']:.6f}")
    print()
    for name, summary in summaries.items():
        print(
            f"{name}: BER={summary['overall_ber']:.6f} ({summary['total_errors']}/{summary['total_bits']}), "
            f"SER={summary['overall_ser']:.6f} ({summary['total_symbol_errors']}/{summary['total_symbols']})"
        )
    print()
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
