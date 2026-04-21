#!/usr/bin/env python3
"""
Offline AWGN impairment experiment on an already validated SDR capture.

This intentionally reuses saved mqam_validation.json/.npz alignment artifacts.
It adds synthetic complex Gaussian noise to the aligned pilot+payload received
symbols, then runs the same detector backends used by receiver_experiment.py.
"""

from __future__ import annotations

import argparse
import importlib
import json
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
    import torch  # noqa: F401

import numpy as np

from core.frame_processing import (
    DetectorFrameInputs,
    PayloadCase,
    build_payload_cases_from_phase_result,
    make_detector,
    run_detector_on_cases,
)
from core.modulation import bits_per_symbol, bits_to_symbols, symbols_to_bits
from qpsk_validate_from_post_costas import (
    estimate_channel,
    estimate_snr_db,
    load_expected_payloads,
    load_metadata,
    pilot_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline AWGN detector experiment on a validated capture.")
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--target-snr-db", type=float, default=5.0)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--detectors", nargs="+", default=["mmse", "mmse_df", "icl", "defined"])
    parser.add_argument("--tx-frame", default="transmitted_frame.bin")
    parser.add_argument("--metadata", default="frame_metadata.json")
    parser.add_argument("--validation-json", default="mqam_validation.json")
    parser.add_argument("--validation-npz", default="mqam_validation.npz")
    parser.add_argument("--icl-module", default="PaperICL")
    parser.add_argument("--icl-callable", default="detect_symbol_with_context")
    parser.add_argument("--defined-module", default="PaperDEFINED")
    parser.add_argument("--defined-callable", default="detect_symbol_with_context")
    parser.add_argument("--output", default="offline_awgn_experiment.json")
    return parser.parse_args()


def load_callable(module_name: str | None, callable_name: str | None):
    if not module_name or not callable_name:
        return None
    module = importlib.import_module(module_name)
    return getattr(module, callable_name)


def load_saved_phase_result(capture_dir: Path, validation_json: str, validation_npz: str) -> dict:
    validation_path = capture_dir / validation_json
    symbols_path = capture_dir / validation_npz
    if not validation_path.exists():
        raise FileNotFoundError(validation_path)
    if not symbols_path.exists():
        raise FileNotFoundError(symbols_path)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    best_phase = validation.get("best_phase")
    if not best_phase:
        raise ValueError(f"{validation_path} does not contain a best_phase result.")
    symbols = np.load(symbols_path, allow_pickle=True)
    result = dict(best_phase)
    result["phase"] = result.get("phase", validation.get("phase"))
    result["decimated_symbols"] = symbols["decimated_symbols"]
    return result


def symbol_error_counts(expected_bits: np.ndarray, detected_bits: np.ndarray, modulation: str) -> tuple[int, int]:
    bits_per = bits_per_symbol(modulation)
    usable = min(len(expected_bits), len(detected_bits))
    usable -= usable % bits_per
    if usable <= 0:
        return 0, 0
    expected_groups = expected_bits[:usable].reshape(-1, bits_per)
    detected_groups = detected_bits[:usable].reshape(-1, bits_per)
    return int(np.sum(np.any(expected_groups != detected_groups, axis=1))), int(len(expected_groups))


def detector_summary(detector, payload_cases: list[PayloadCase]) -> dict:
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


def add_awgn_to_cases(
    payload_cases: list[PayloadCase],
    target_snr_db: float,
    rng: np.random.Generator,
) -> tuple[list[PayloadCase], list[dict]]:
    noisy_cases: list[PayloadCase] = []
    noise_rows: list[dict] = []
    target_linear = 10.0 ** (target_snr_db / 10.0)

    for case in payload_cases:
        frame_inputs = case.frame_inputs
        modulation = frame_inputs.config.modulation
        pilot_tx = frame_inputs.pilot_tx_symbols.astype(np.complex64)
        data_tx = bits_to_symbols(case.expected_bits, modulation).astype(np.complex64)
        tx_symbols = np.concatenate([pilot_tx, data_tx]).astype(np.complex64)
        clean_rx = np.concatenate([frame_inputs.pilot_rx, frame_inputs.data_rx]).astype(np.complex64)

        h_hat = estimate_channel(clean_rx, tx_symbols)
        signal = h_hat * tx_symbols
        signal_power = float(np.mean(np.abs(signal) ** 2))
        noise_power = signal_power / target_linear
        noise = np.sqrt(noise_power / 2.0) * (
            rng.standard_normal(len(clean_rx)) + 1j * rng.standard_normal(len(clean_rx))
        )
        noise = noise.astype(np.complex64)
        noisy_rx = (clean_rx + noise).astype(np.complex64)
        actual_noise_power = float(np.mean(np.abs(noise) ** 2))
        actual_snr_db = float(10.0 * np.log10(max(signal_power, 1e-12) / max(actual_noise_power, 1e-12)))

        pilot_count = len(frame_inputs.pilot_rx)
        noisy_pilot = noisy_rx[:pilot_count]
        noisy_data = noisy_rx[pilot_count:]
        noisy_h = estimate_channel(noisy_pilot, pilot_tx)
        noisy_eq_pilot = noisy_pilot / (noisy_h + 1e-12)
        noisy_pilot_bits = symbols_to_bits(noisy_eq_pilot, modulation)
        noisy_pilot_ber = float(np.mean(noisy_pilot_bits != frame_inputs.pilot_tx_bits))

        noisy_cases.append(
            PayloadCase(
                payload_index=case.payload_index,
                expected_payload_index=case.expected_payload_index,
                frame_inputs=DetectorFrameInputs(
                    config=frame_inputs.config,
                    pilot_rx=noisy_pilot,
                    pilot_tx_bits=frame_inputs.pilot_tx_bits.copy(),
                    data_rx=noisy_data,
                ),
                expected_bits=case.expected_bits.copy(),
                pilot_score=pilot_score(noisy_pilot, pilot_tx),
                pilot_ber=noisy_pilot_ber,
                pilot_drift=case.pilot_drift,
            )
        )
        noise_rows.append(
            {
                "payload_index": case.payload_index,
                "target_snr_db": target_snr_db,
                "actual_synthetic_snr_db": actual_snr_db,
                "pilot_estimated_snr_db": estimate_snr_db(noisy_pilot, pilot_tx, noisy_h),
                "signal_power": signal_power,
                "synthetic_noise_power": actual_noise_power,
                "pilot_ber_after_noise": noisy_pilot_ber,
                "pilot_score_after_noise": pilot_score(noisy_pilot, pilot_tx),
            }
        )

    return noisy_cases, noise_rows


def summarize_repetitions(rows: list[dict], detector_names: list[str]) -> dict:
    out = {}
    for name in detector_names:
        ber_values = np.array([row["detectors"][name]["ber"] for row in rows], dtype=np.float64)
        ser_values = np.array([row["detectors"][name]["ser"] for row in rows], dtype=np.float64)
        bit_errors = int(sum(row["detectors"][name]["bit_errors"] for row in rows))
        bits = int(sum(row["detectors"][name]["bits"] for row in rows))
        sym_errors = int(sum(row["detectors"][name]["symbol_errors"] for row in rows))
        symbols = int(sum(row["detectors"][name]["symbols"] for row in rows))
        out[name] = {
            "ber_mean": float(np.mean(ber_values)) if len(ber_values) else 1.0,
            "ber_std": float(np.std(ber_values)) if len(ber_values) else 0.0,
            "ber_pooled": bit_errors / bits if bits else 1.0,
            "ser_mean": float(np.mean(ser_values)) if len(ser_values) else 1.0,
            "ser_std": float(np.std(ser_values)) if len(ser_values) else 0.0,
            "ser_pooled": sym_errors / symbols if symbols else 1.0,
            "bit_errors": bit_errors,
            "bits": bits,
            "symbol_errors": sym_errors,
            "symbols": symbols,
        }
    return out


def main() -> int:
    args = parse_args()
    capture_dir = Path(args.capture_dir)
    metadata = load_metadata(capture_dir / args.metadata)
    tx_bits = np.fromfile(capture_dir / args.tx_frame, dtype=np.uint8)
    expected_payload_bits = load_expected_payloads(metadata, capture_dir, tx_bits)
    phase_result = load_saved_phase_result(capture_dir, args.validation_json, args.validation_npz)
    payload_cases = build_payload_cases_from_phase_result(
        phase_result=phase_result,
        decimated_symbols=phase_result["decimated_symbols"],
        metadata=metadata,
        tx_bits=tx_bits,
        expected_payload_bits=expected_payload_bits,
    )
    if not payload_cases:
        raise RuntimeError("No payload cases could be constructed from saved validation artifacts.")

    detector_names = [name.lower() for name in args.detectors]
    icl_callable = load_callable(args.icl_module, args.icl_callable) if "icl" in detector_names else None
    defined_callable = load_callable(args.defined_module, args.defined_callable) if "defined" in detector_names else None
    config = payload_cases[0].frame_inputs.config
    rng = np.random.default_rng(args.seed)

    clean_summaries = {}
    for detector_name in detector_names:
        detector = make_detector(
            detector_name,
            config,
            defined_model_func=defined_callable,
            icl_model_func=icl_callable,
        )
        clean_summaries[detector_name] = detector_summary(detector, payload_cases)

    repetition_rows = []
    for rep in range(args.repetitions):
        noisy_cases, noise_rows = add_awgn_to_cases(payload_cases, args.target_snr_db, rng)
        detector_rows = {}
        for detector_name in detector_names:
            detector = make_detector(
                detector_name,
                config,
                defined_model_func=defined_callable,
                icl_model_func=icl_callable,
            )
            detector_rows[detector_name] = detector_summary(detector, noisy_cases)
        repetition_rows.append(
            {
                "repetition": rep,
                "actual_synthetic_snr_db_mean": float(np.mean([row["actual_synthetic_snr_db"] for row in noise_rows])),
                "pilot_estimated_snr_db_mean": float(np.mean([row["pilot_estimated_snr_db"] for row in noise_rows])),
                "mean_pilot_ber_after_noise": float(np.mean([row["pilot_ber_after_noise"] for row in noise_rows])),
                "noise": noise_rows,
                "detectors": detector_rows,
            }
        )

    aggregate = summarize_repetitions(repetition_rows, detector_names)
    output = {
        "capture_dir": str(capture_dir),
        "modulation": metadata.get("modulation"),
        "pilot_symbols": metadata.get("pilot_symbols"),
        "data_symbols": metadata.get("data_symbols"),
        "target_snr_db": args.target_snr_db,
        "repetitions": args.repetitions,
        "seed": args.seed,
        "saved_alignment": {
            "phase": phase_result.get("phase"),
            "lock_score": phase_result.get("lock_score"),
            "clean_validation_ber": phase_result.get("overall_ber"),
            "clean_validation_mean_snr_db": phase_result.get("mean_snr_db"),
        },
        "clean_detectors": clean_summaries,
        "aggregate_noisy_detectors": aggregate,
        "repetition_results": repetition_rows,
    }
    output_path = capture_dir / args.output
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    actual_snr_values = [row["actual_synthetic_snr_db_mean"] for row in repetition_rows]
    pilot_snr_values = [row["pilot_estimated_snr_db_mean"] for row in repetition_rows]
    print("=" * 72)
    print("OFFLINE AWGN DETECTOR EXPERIMENT")
    print("=" * 72)
    print(f"Capture dir:       {capture_dir}")
    print(f"Modulation:        {metadata.get('modulation')}")
    print(f"Pilot symbols:     {metadata.get('pilot_symbols')}")
    print(f"Target SNR:        {args.target_snr_db:.2f} dB")
    print(f"Actual AWGN SNR:   {np.mean(actual_snr_values):.2f} +/- {np.std(actual_snr_values):.2f} dB")
    print(f"Pilot-est SNR:     {np.mean(pilot_snr_values):.2f} +/- {np.std(pilot_snr_values):.2f} dB")
    print(f"Repetitions:       {args.repetitions}")
    print()
    print("Clean detector results:")
    for name, summary in clean_summaries.items():
        print(f"  {name}: BER={summary['ber']:.6f}, SER={summary['ser']:.6f}")
    print()
    print("Noisy aggregate results:")
    for name, summary in aggregate.items():
        print(
            f"  {name}: BER={summary['ber_mean']:.6f} +/- {summary['ber_std']:.6f} "
            f"(pooled={summary['ber_pooled']:.6f}), "
            f"SER={summary['ser_mean']:.6f} +/- {summary['ser_std']:.6f}"
        )
    print()
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
