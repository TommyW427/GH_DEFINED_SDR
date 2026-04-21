#!/usr/bin/env python3
"""
Run a paper-style BER-vs-pilot-count sweep for SDR captures.

For each pilot count k, this script:
  1. generates a paper-style frame with total length T=31 symbols
  2. runs the existing headless SDR capture/validation harness
  3. optionally runs the modular detector experiment
  4. aggregates BER vs pilot count across detectors
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, pstdev

from generate_defined_paper_frame import generate_defined_paper_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DEFINED-style BER-vs-pilots SDR sweep.")
    parser.add_argument("--modulation", required=True, choices=["BPSK", "QPSK", "16QAM", "64QAM"])
    parser.add_argument("--pilot-symbols", nargs="+", type=int, default=[1, 2, 4, 8, 16, 24, 30])
    parser.add_argument("--total-symbols", type=int, default=31)
    parser.add_argument("--preamble-symbols", type=int, default=6000)
    parser.add_argument("--captures-dir", default="captures")
    parser.add_argument("--results-dir", default="experiments")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--rx-device-args", default="driver=plutosdr")
    parser.add_argument("--tx-device-args", default="driver=plutosdr")
    parser.add_argument("--rx-time", type=float, default=6.0)
    parser.add_argument("--tx-delay", type=float, default=1.5)
    parser.add_argument("--tx-time", type=float, default=None)
    parser.add_argument("--inter-run-delay", type=float, default=1.0)
    parser.add_argument("--rx-gain", type=float, default=40.0)
    parser.add_argument("--carrier-recovery", choices=["none", "costas"], default="none")
    parser.add_argument("--costas-bw", type=float, default=0.001)
    parser.add_argument("--tx-gain", type=float, default=50.0)
    parser.add_argument("--tx-scale", type=float, default=0.2)
    parser.add_argument("--detectors", nargs="+", default=["mmse", "mmse_df"])
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--icl-checkpoint", default=None)
    parser.add_argument("--defined-checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lock-threshold", type=float, default=0.95)
    parser.add_argument(
        "--exclude-bad-lock",
        action="store_true",
        help="Exclude runs below --lock-threshold from primary aggregate BER/SER statistics.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def path_for_cli(path: Path, root: Path) -> str:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    try:
        return str(resolved_path.relative_to(resolved_root))
    except ValueError:
        return os.path.relpath(resolved_path, resolved_root)


def summarize_metric(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    if args.modulation in {"16QAM", "64QAM"} and args.carrier_recovery == "costas":
        raise ValueError(
            f"--carrier-recovery costas is not supported for {args.modulation} in the current "
            "M-QAM receiver path. Use --carrier-recovery none."
        )
    if args.modulation in {"16QAM", "64QAM"}:
        if args.rx_time == 6.0:
            args.rx_time = 10.0
        if args.tx_delay == 1.5:
            args.tx_delay = 2.0
        if args.tx_time is None:
            args.tx_time = 6.0
        if args.inter_run_delay == 1.0:
            args.inter_run_delay = 2.0
    sweep_tag = args.tag or time.strftime(f"defined_pilot_sweep_%Y%m%d_%H%M%S")
    results_dir = (root / args.results_dir / sweep_tag).resolve()
    results_dir.mkdir(parents=True, exist_ok=False)

    all_results = []
    for pilot_symbols in args.pilot_symbols:
        if pilot_symbols >= args.total_symbols:
            raise ValueError(f"pilot_symbols={pilot_symbols} must be smaller than total_symbols={args.total_symbols}")

        frame_path, metadata_path, payload_path = generate_defined_paper_frame(
            modulation=args.modulation,
            pilot_symbols=pilot_symbols,
            total_symbols=args.total_symbols,
            preamble_symbols=args.preamble_symbols,
            seed=args.seed,
        )

        repetition_rows = []
        for repetition in range(args.repetitions):
            run_tag = f"{sweep_tag}_{args.modulation.lower()}_paper_k{pilot_symbols:02d}_r{repetition + 1:02d}"
            cmd = [
                args.python,
                str(root / "run_mqam_headless_test.py"),
                "--python",
                args.python,
                "--modulation",
                args.modulation,
                "--tx-frame",
                path_for_cli(frame_path, root),
                "--metadata",
                path_for_cli(metadata_path, root),
                "--captures-dir",
                args.captures_dir,
                "--tag",
                run_tag,
                "--rx-device-args",
                args.rx_device_args,
                "--tx-device-args",
                args.tx_device_args,
                "--rx-time",
                str(args.rx_time),
                "--tx-delay",
                str(args.tx_delay),
                "--rx-gain",
                str(args.rx_gain),
                "--carrier-recovery",
                args.carrier_recovery,
                "--costas-bw",
                str(args.costas_bw),
                "--tx-gain",
                str(args.tx_gain),
                "--tx-scale",
                str(args.tx_scale),
                "--run-experiment",
                "--experiment-detectors",
                *args.detectors,
            ]
            if args.icl_checkpoint:
                cmd.extend(["--icl-checkpoint", args.icl_checkpoint])
            if args.defined_checkpoint:
                cmd.extend(["--defined-checkpoint", args.defined_checkpoint])
            if args.tx_time is not None:
                cmd.extend(["--tx-time", str(args.tx_time)])

            print()
            print("=" * 72)
            print(f"PILOT SWEEP: k={pilot_symbols} / T={args.total_symbols} / rep={repetition + 1}/{args.repetitions}")
            print("=" * 72)
            run = subprocess.run(cmd, cwd=root, text=True, capture_output=True)
            sys.stdout.write(run.stdout)
            if run.stderr:
                sys.stderr.write(run.stderr)
            if run.returncode != 0:
                raise RuntimeError(
                    f"Pilot sweep run failed for k={pilot_symbols}, repetition={repetition + 1} "
                    f"with exit code {run.returncode}"
                )
            if args.inter_run_delay > 0 and repetition + 1 < args.repetitions:
                print(f"Settling for {args.inter_run_delay:.1f} s before next repetition...")
                time.sleep(args.inter_run_delay)

            capture_dir = root / args.captures_dir / run_tag
            validation_summary = load_json(capture_dir / "mqam_validation.json")
            experiment_summary = load_json(capture_dir / "receiver_experiment.json")

            best_phase = validation_summary["best_phase"]
            repetition_row = {
                "pilot_symbols": pilot_symbols,
                "repetition": repetition + 1,
                "total_symbols": args.total_symbols,
                "payload_symbols": args.total_symbols - pilot_symbols,
                "capture_dir": str(capture_dir),
                "lock_score": best_phase["lock_score"],
                "good_lock": bool(best_phase["lock_score"] >= args.lock_threshold),
                "included": bool(best_phase["lock_score"] >= args.lock_threshold) if args.exclude_bad_lock else True,
                "validation_ber": best_phase["overall_ber"],
                "validation_mean_pilot_ber": best_phase["mean_pilot_ber"],
                "validation_mean_snr_db": best_phase.get("mean_snr_db"),
            }
            for detector_name, detector_result in experiment_summary["detectors"].items():
                repetition_row[f"{detector_name}_ber"] = detector_result["overall_ber"]
                repetition_row[f"{detector_name}_ser"] = detector_result.get("overall_ser", detector_result["overall_ber"])
            repetition_rows.append(repetition_row)

        aggregate_row = {
            "pilot_symbols": pilot_symbols,
            "total_symbols": args.total_symbols,
            "payload_symbols": args.total_symbols - pilot_symbols,
            "repetitions": args.repetitions,
            "lock_threshold": args.lock_threshold,
            "exclude_bad_lock": args.exclude_bad_lock,
            "good_lock_runs": int(sum(1 for row in repetition_rows if row["good_lock"])),
            "included_runs": int(sum(1 for row in repetition_rows if row["included"])),
            "sync_success_rate": float(sum(1 for row in repetition_rows if row["good_lock"]) / len(repetition_rows)),
            "runs": repetition_rows,
        }
        included_rows = [row for row in repetition_rows if row["included"]]
        aggregate_row["lock_score"] = summarize_metric([row["lock_score"] for row in included_rows])
        aggregate_row["validation_ber"] = summarize_metric([row["validation_ber"] for row in included_rows])
        aggregate_row["validation_mean_pilot_ber"] = summarize_metric([row["validation_mean_pilot_ber"] for row in included_rows])
        aggregate_row["validation_mean_snr_db"] = summarize_metric([row["validation_mean_snr_db"] for row in included_rows])
        aggregate_row["all_run_lock_score"] = summarize_metric([row["lock_score"] for row in repetition_rows])
        aggregate_row["all_run_validation_ber"] = summarize_metric([row["validation_ber"] for row in repetition_rows])
        aggregate_row["all_run_validation_mean_snr_db"] = summarize_metric([row["validation_mean_snr_db"] for row in repetition_rows])
        good_lock_rows = [row for row in repetition_rows if row["good_lock"]]
        aggregate_row["good_lock_validation_ber"] = summarize_metric([row["validation_ber"] for row in good_lock_rows])
        aggregate_row["good_lock_validation_mean_snr_db"] = summarize_metric([row["validation_mean_snr_db"] for row in good_lock_rows])
        detector_summaries = {}
        for detector_name in args.detectors:
            detector_summaries[detector_name] = {
                "ber": summarize_metric([row[f"{detector_name}_ber"] for row in included_rows]),
                "ser": summarize_metric([row[f"{detector_name}_ser"] for row in included_rows]),
                "all_run_ber": summarize_metric([row[f"{detector_name}_ber"] for row in repetition_rows]),
                "all_run_ser": summarize_metric([row[f"{detector_name}_ser"] for row in repetition_rows]),
                "good_lock_ber": summarize_metric([row[f"{detector_name}_ber"] for row in good_lock_rows]),
                "good_lock_ser": summarize_metric([row[f"{detector_name}_ser"] for row in good_lock_rows]),
            }
        aggregate_row["detectors"] = detector_summaries
        all_results.append(aggregate_row)

    summary = {
        "modulation": args.modulation,
        "total_symbols": args.total_symbols,
        "pilot_symbols_tested": args.pilot_symbols,
        "detectors": args.detectors,
        "repetitions": args.repetitions,
        "lock_threshold": args.lock_threshold,
        "exclude_bad_lock": args.exclude_bad_lock,
        "rx_time": args.rx_time,
        "tx_delay": args.tx_delay,
        "tx_time": args.tx_time,
        "inter_run_delay": args.inter_run_delay,
        "results": all_results,
    }

    summary_path = results_dir / "defined_pilot_sweep_summary.json"
    csv_path = results_dir / "defined_pilot_sweep_summary.csv"
    detailed_csv_path = results_dir / "defined_pilot_sweep_detailed.csv"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fieldnames = [
        "pilot_symbols",
        "total_symbols",
        "payload_symbols",
        "repetitions",
        "good_lock_runs",
        "included_runs",
        "sync_success_rate",
        "lock_score_mean",
        "lock_score_std",
        "validation_ber_mean",
        "validation_ber_std",
        "validation_mean_pilot_ber_mean",
        "validation_mean_pilot_ber_std",
        "validation_mean_snr_db_mean",
        "validation_mean_snr_db_std",
        "good_lock_validation_ber_mean",
        "good_lock_validation_mean_snr_db_mean",
        "all_run_lock_score_mean",
        "all_run_validation_ber_mean",
        "all_run_validation_mean_snr_db_mean",
        *[f"{name}_ber_mean" for name in args.detectors],
        *[f"{name}_ber_std" for name in args.detectors],
        *[f"{name}_ser_mean" for name in args.detectors],
        *[f"{name}_ser_std" for name in args.detectors],
        *[f"{name}_all_run_ber_mean" for name in args.detectors],
        *[f"{name}_all_run_ser_mean" for name in args.detectors],
        *[f"{name}_good_lock_ber_mean" for name in args.detectors],
        *[f"{name}_good_lock_ser_mean" for name in args.detectors],
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            flat_row = {
                "pilot_symbols": row["pilot_symbols"],
                "total_symbols": row["total_symbols"],
                "payload_symbols": row["payload_symbols"],
                "repetitions": row["repetitions"],
                "good_lock_runs": row["good_lock_runs"],
                "included_runs": row["included_runs"],
                "sync_success_rate": row["sync_success_rate"],
                "lock_score_mean": row["lock_score"]["mean"],
                "lock_score_std": row["lock_score"]["std"],
                "validation_ber_mean": row["validation_ber"]["mean"],
                "validation_ber_std": row["validation_ber"]["std"],
                "validation_mean_pilot_ber_mean": row["validation_mean_pilot_ber"]["mean"],
                "validation_mean_pilot_ber_std": row["validation_mean_pilot_ber"]["std"],
                "validation_mean_snr_db_mean": row["validation_mean_snr_db"]["mean"],
                "validation_mean_snr_db_std": row["validation_mean_snr_db"]["std"],
                "good_lock_validation_ber_mean": row["good_lock_validation_ber"]["mean"],
                "good_lock_validation_mean_snr_db_mean": row["good_lock_validation_mean_snr_db"]["mean"],
                "all_run_lock_score_mean": row["all_run_lock_score"]["mean"],
                "all_run_validation_ber_mean": row["all_run_validation_ber"]["mean"],
                "all_run_validation_mean_snr_db_mean": row["all_run_validation_mean_snr_db"]["mean"],
            }
            for detector_name in args.detectors:
                flat_row[f"{detector_name}_ber_mean"] = row["detectors"][detector_name]["ber"]["mean"]
                flat_row[f"{detector_name}_ber_std"] = row["detectors"][detector_name]["ber"]["std"]
                flat_row[f"{detector_name}_ser_mean"] = row["detectors"][detector_name]["ser"]["mean"]
                flat_row[f"{detector_name}_ser_std"] = row["detectors"][detector_name]["ser"]["std"]
                flat_row[f"{detector_name}_all_run_ber_mean"] = row["detectors"][detector_name]["all_run_ber"]["mean"]
                flat_row[f"{detector_name}_all_run_ser_mean"] = row["detectors"][detector_name]["all_run_ser"]["mean"]
                flat_row[f"{detector_name}_good_lock_ber_mean"] = row["detectors"][detector_name]["good_lock_ber"]["mean"]
                flat_row[f"{detector_name}_good_lock_ser_mean"] = row["detectors"][detector_name]["good_lock_ser"]["mean"]
            writer.writerow(flat_row)

    detail_fieldnames = [
        "pilot_symbols",
        "repetition",
        "total_symbols",
        "payload_symbols",
        "lock_score",
        "good_lock",
        "included",
        "validation_ber",
        "validation_mean_pilot_ber",
        "validation_mean_snr_db",
        *[f"{name}_ber" for name in args.detectors],
        *[f"{name}_ser" for name in args.detectors],
        "capture_dir",
    ]
    with detailed_csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=detail_fieldnames)
        writer.writeheader()
        for row in all_results:
            for run in row["runs"]:
                writer.writerow(run)

    print()
    print("=" * 72)
    print("DEFINED PILOT SWEEP COMPLETE")
    print("=" * 72)
    print(f"Summary JSON: {summary_path}")
    print(f"Summary CSV:  {csv_path}")
    print(f"Detail CSV:   {detailed_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
