#!/usr/bin/env python3
"""
Run a paper-style pilot sweep across two computers.

Launch this on the TX/controller computer. Each run starts the RX flowgraph on
the remote computer over SSH, transmits locally, copies the capture back, and
aggregates validation/detector metrics.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SRC_DIR))

from generate_defined_paper_frame import generate_defined_paper_frame  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run distributed DEFINED-style BER-vs-pilots SDR sweep.")
    parser.add_argument("--modulation", required=True, choices=["BPSK", "QPSK", "16QAM", "64QAM"])
    parser.add_argument("--pilot-symbols", nargs="+", type=int, default=[1, 2, 4, 8, 16, 24, 30])
    parser.add_argument("--total-symbols", type=int, default=31)
    parser.add_argument("--preamble-symbols", type=int, default=6000)
    parser.add_argument("--sync-tail-symbols", type=int, default=64)
    parser.add_argument("--remote-rx-host", required=True)
    parser.add_argument("--remote-repo", default=str(REPO_ROOT))
    parser.add_argument("--remote-python", default=sys.executable)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--ssh-port", type=int, default=None)
    parser.add_argument("--ssh-option", action="append", default=[])
    parser.add_argument("--captures-dir", default="captures")
    parser.add_argument("--results-dir", default="experiments")
    parser.add_argument("--tag", default=None)
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
    parser.add_argument("--exclude-bad-lock", action="store_true")
    parser.add_argument("--keep-remote-capture", action="store_true")
    return parser.parse_args()


def local_env() -> dict[str, str]:
    env = os.environ.copy()
    py_path = f"{SRC_DIR}{os.pathsep}{SCRIPTS_DIR}"
    env["PYTHONPATH"] = py_path if not env.get("PYTHONPATH") else f"{py_path}{os.pathsep}{env['PYTHONPATH']}"
    return env


def summarize_metric(values: list[float | None]) -> dict:
    clean_values = [float(value) for value in values if value is not None]
    if not clean_values:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": mean(clean_values),
        "std": pstdev(clean_values) if len(clean_values) > 1 else 0.0,
        "min": min(clean_values),
        "max": max(clean_values),
    }


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    if args.modulation in {"16QAM", "64QAM"} and args.carrier_recovery == "costas":
        raise ValueError(f"--carrier-recovery costas is not supported for {args.modulation}. Use --carrier-recovery none.")

    sweep_tag = args.tag or time.strftime(f"distributed_pilot_sweep_%Y%m%d_%H%M%S")
    results_dir = (REPO_ROOT / args.results_dir / sweep_tag).resolve()
    results_dir.mkdir(parents=True, exist_ok=False)

    all_results = []
    for pilot_symbols in args.pilot_symbols:
        if pilot_symbols >= args.total_symbols:
            raise ValueError(f"pilot_symbols={pilot_symbols} must be smaller than total_symbols={args.total_symbols}")

        modulation_tag = args.modulation.lower()
        frame_path = REPO_ROOT / "frames" / f"distributed_{modulation_tag}_k{pilot_symbols:02d}.bin"
        payload_path = REPO_ROOT / f"payloads_{modulation_tag}" / f"distributed_payload_k{pilot_symbols:02d}.bin"
        frame_path, metadata_path, _ = generate_defined_paper_frame(
            modulation=args.modulation,
            pilot_symbols=pilot_symbols,
            total_symbols=args.total_symbols,
            preamble_symbols=args.preamble_symbols,
            sync_tail_symbols=args.sync_tail_symbols,
            seed=args.seed,
            frame_output=frame_path,
            payload_output=payload_path,
        )

        repetition_rows = []
        for repetition in range(args.repetitions):
            run_tag = f"{sweep_tag}_{modulation_tag}_k{pilot_symbols:02d}_r{repetition + 1:02d}"
            cmd = [
                args.python,
                str(SCRIPTS_DIR / "run_distributed_mqam_test.py"),
                "--python",
                args.python,
                "--remote-python",
                args.remote_python,
                "--modulation",
                args.modulation,
                "--tx-frame",
                str(frame_path),
                "--metadata",
                str(metadata_path),
                "--remote-rx-host",
                args.remote_rx_host,
                "--remote-repo",
                args.remote_repo,
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
            if args.ssh_port is not None:
                cmd.extend(["--ssh-port", str(args.ssh_port)])
            for ssh_option in args.ssh_option:
                cmd.extend(["--ssh-option", ssh_option])
            if args.tx_time is not None:
                cmd.extend(["--tx-time", str(args.tx_time)])
            if args.icl_checkpoint:
                cmd.extend(["--icl-checkpoint", args.icl_checkpoint])
            if args.defined_checkpoint:
                cmd.extend(["--defined-checkpoint", args.defined_checkpoint])
            if args.keep_remote_capture:
                cmd.append("--keep-remote-capture")

            print()
            print("=" * 72)
            print(f"DISTRIBUTED PILOT SWEEP: k={pilot_symbols} / T={args.total_symbols} / rep={repetition + 1}/{args.repetitions}")
            print("=" * 72)
            run = subprocess.run(cmd, cwd=REPO_ROOT, env=local_env(), text=True, capture_output=True)
            if run.stdout:
                sys.stdout.write(run.stdout)
            if run.stderr:
                sys.stderr.write(run.stderr)
            if run.returncode != 0:
                raise RuntimeError(
                    f"Distributed sweep run failed for k={pilot_symbols}, repetition={repetition + 1}, "
                    f"exit code {run.returncode}"
                )
            if args.inter_run_delay > 0 and repetition + 1 < args.repetitions:
                print(f"Settling for {args.inter_run_delay:.1f} s before next repetition...")
                time.sleep(args.inter_run_delay)

            capture_dir = REPO_ROOT / args.captures_dir / run_tag
            validation_summary = load_json(capture_dir / "mqam_validation.json")
            experiment_summary = load_json(capture_dir / "receiver_experiment.json")
            best_phase = validation_summary["best_phase"]
            lock_score = best_phase["lock_score"]
            repetition_row = {
                "pilot_symbols": pilot_symbols,
                "repetition": repetition + 1,
                "total_symbols": args.total_symbols,
                "payload_symbols": args.total_symbols - pilot_symbols,
                "capture_dir": str(capture_dir),
                "lock_score": lock_score,
                "good_lock": bool(lock_score >= args.lock_threshold),
                "included": bool(lock_score >= args.lock_threshold) if args.exclude_bad_lock else True,
                "validation_ber": best_phase["overall_ber"],
                "validation_mean_pilot_ber": best_phase["mean_pilot_ber"],
                "validation_mean_snr_db": best_phase.get("mean_snr_db"),
            }
            for detector_name, detector_result in experiment_summary["detectors"].items():
                repetition_row[f"{detector_name}_ber"] = detector_result["overall_ber"]
                repetition_row[f"{detector_name}_ser"] = detector_result.get("overall_ser", detector_result["overall_ber"])
            repetition_rows.append(repetition_row)

        included_rows = [row for row in repetition_rows if row["included"]]
        good_lock_rows = [row for row in repetition_rows if row["good_lock"]]
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
        all_results.append(
            {
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
                "lock_score": summarize_metric([row["lock_score"] for row in included_rows]),
                "validation_ber": summarize_metric([row["validation_ber"] for row in included_rows]),
                "validation_mean_pilot_ber": summarize_metric([row["validation_mean_pilot_ber"] for row in included_rows]),
                "validation_mean_snr_db": summarize_metric([row["validation_mean_snr_db"] for row in included_rows]),
                "all_run_lock_score": summarize_metric([row["lock_score"] for row in repetition_rows]),
                "all_run_validation_ber": summarize_metric([row["validation_ber"] for row in repetition_rows]),
                "all_run_validation_mean_snr_db": summarize_metric([row["validation_mean_snr_db"] for row in repetition_rows]),
                "good_lock_validation_ber": summarize_metric([row["validation_ber"] for row in good_lock_rows]),
                "good_lock_validation_mean_snr_db": summarize_metric([row["validation_mean_snr_db"] for row in good_lock_rows]),
                "detectors": detector_summaries,
            }
        )

    summary = {
        "mode": "distributed_two_computer",
        "remote_rx_host": args.remote_rx_host,
        "remote_repo": args.remote_repo,
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

    summary_path = results_dir / "distributed_pilot_sweep_summary.json"
    csv_path = results_dir / "distributed_pilot_sweep_summary.csv"
    detailed_csv_path = results_dir / "distributed_pilot_sweep_detailed.csv"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary_fields = [
        "pilot_symbols",
        "total_symbols",
        "payload_symbols",
        "repetitions",
        "good_lock_runs",
        "included_runs",
        "sync_success_rate",
        "lock_score_mean",
        "validation_ber_mean",
        "validation_mean_pilot_ber_mean",
        "validation_mean_snr_db_mean",
        "all_run_lock_score_mean",
        "all_run_validation_ber_mean",
        "all_run_validation_mean_snr_db_mean",
        *[f"{name}_ber_mean" for name in args.detectors],
        *[f"{name}_ser_mean" for name in args.detectors],
        *[f"{name}_all_run_ber_mean" for name in args.detectors],
        *[f"{name}_all_run_ser_mean" for name in args.detectors],
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=summary_fields)
        writer.writeheader()
        for row in all_results:
            flat = {
                "pilot_symbols": row["pilot_symbols"],
                "total_symbols": row["total_symbols"],
                "payload_symbols": row["payload_symbols"],
                "repetitions": row["repetitions"],
                "good_lock_runs": row["good_lock_runs"],
                "included_runs": row["included_runs"],
                "sync_success_rate": row["sync_success_rate"],
                "lock_score_mean": row["lock_score"]["mean"],
                "validation_ber_mean": row["validation_ber"]["mean"],
                "validation_mean_pilot_ber_mean": row["validation_mean_pilot_ber"]["mean"],
                "validation_mean_snr_db_mean": row["validation_mean_snr_db"]["mean"],
                "all_run_lock_score_mean": row["all_run_lock_score"]["mean"],
                "all_run_validation_ber_mean": row["all_run_validation_ber"]["mean"],
                "all_run_validation_mean_snr_db_mean": row["all_run_validation_mean_snr_db"]["mean"],
            }
            for detector_name in args.detectors:
                flat[f"{detector_name}_ber_mean"] = row["detectors"][detector_name]["ber"]["mean"]
                flat[f"{detector_name}_ser_mean"] = row["detectors"][detector_name]["ser"]["mean"]
                flat[f"{detector_name}_all_run_ber_mean"] = row["detectors"][detector_name]["all_run_ber"]["mean"]
                flat[f"{detector_name}_all_run_ser_mean"] = row["detectors"][detector_name]["all_run_ser"]["mean"]
            writer.writerow(flat)

    detail_fields = [
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
        writer = csv.DictWriter(fh, fieldnames=detail_fields)
        writer.writeheader()
        for row in all_results:
            for run in row["runs"]:
                writer.writerow(run)

    print()
    print("=" * 72)
    print("DISTRIBUTED PILOT SWEEP COMPLETE")
    print("=" * 72)
    print(f"Summary JSON: {summary_path}")
    print(f"Summary CSV:  {csv_path}")
    print(f"Detail CSV:   {detailed_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
