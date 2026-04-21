#!/usr/bin/env python3
"""
Run a paper-style pilot sweep across two computers.

Launch this on the RX/controller computer by default. Each run starts the local
RX flowgraph, transmits from the remote TX computer over SSH, and aggregates
validation/detector metrics.
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
    parser.add_argument("--remote-host", default=None)
    parser.add_argument("--remote-tx-host", default=None)
    parser.add_argument("--remote-rx-host", default=None)
    parser.add_argument("--remote-role", choices=["tx", "rx"], default="tx")
    parser.add_argument("--remote-os", choices=["windows", "posix"], default="windows")
    parser.add_argument("--remote-repo", default="C:/Users/tpdub/Downloads/Wireless/GH_DEFINED_SDR")
    parser.add_argument("--remote-python", default="python")
    parser.add_argument("--remote-conda-env", default="gnuradio")
    parser.add_argument("--remote-conda-exe", default="conda")
    parser.add_argument("--remote-sdr-preflight", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument("--detectors", nargs="+", default=["mmse", "mmse_df", "icl", "defined"])
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--icl-checkpoint", default=None)
    parser.add_argument("--defined-checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lock-threshold", type=float, default=0.95)
    parser.add_argument(
        "--retry-until-lock",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Repeat each logical repetition until lock_score reaches --lock-threshold.",
    )
    parser.add_argument(
        "--max-lock-attempts",
        type=int,
        default=5,
        help="Maximum attempts per logical repetition when --retry-until-lock is enabled. Use 0 for unlimited.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=1.0,
        help="Delay between bad-lock retry attempts.",
    )
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


def resolve_remote_host(args: argparse.Namespace) -> str:
    host = args.remote_host
    if args.remote_role == "tx":
        host = host or args.remote_tx_host
    else:
        host = host or args.remote_rx_host
    if not host:
        raise ValueError("Provide --remote-host, --remote-tx-host, or --remote-rx-host.")
    return host


def main() -> int:
    args = parse_args()
    remote_host = resolve_remote_host(args)
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
        attempt_rows = []
        for repetition in range(args.repetitions):
            selected_row = None
            attempt = 1
            while True:
                if args.retry_until_lock:
                    if args.max_lock_attempts > 0 and attempt > args.max_lock_attempts:
                        break
                    run_tag = f"{sweep_tag}_{modulation_tag}_k{pilot_symbols:02d}_r{repetition + 1:02d}_a{attempt:02d}"
                else:
                    if attempt > 1:
                        break
                    run_tag = f"{sweep_tag}_{modulation_tag}_k{pilot_symbols:02d}_r{repetition + 1:02d}"

                cmd = [
                    args.python,
                    str(SCRIPTS_DIR / "run_distributed_mqam_test.py"),
                    "--python",
                    args.python,
                    "--remote-python",
                    args.remote_python,
                    "--remote-conda-env",
                    args.remote_conda_env,
                    "--remote-conda-exe",
                    args.remote_conda_exe,
                    "--remote-sdr-preflight" if args.remote_sdr_preflight else "--no-remote-sdr-preflight",
                    "--modulation",
                    args.modulation,
                    "--tx-frame",
                    str(frame_path),
                    "--metadata",
                    str(metadata_path),
                    "--remote-host",
                    remote_host,
                    "--remote-role",
                    args.remote_role,
                    "--remote-os",
                    args.remote_os,
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
                print(
                    f"DISTRIBUTED PILOT SWEEP: k={pilot_symbols} / T={args.total_symbols} / "
                    f"rep={repetition + 1}/{args.repetitions} / attempt={attempt}"
                )
                print("=" * 72)
                run = subprocess.run(cmd, cwd=REPO_ROOT, env=local_env(), text=True, capture_output=True)
                if run.stdout:
                    sys.stdout.write(run.stdout)
                if run.stderr:
                    sys.stderr.write(run.stderr)
                if run.returncode != 0:
                    raise RuntimeError(
                        f"Distributed sweep run failed for k={pilot_symbols}, repetition={repetition + 1}, "
                        f"attempt={attempt}, exit code {run.returncode}"
                    )

                capture_dir = REPO_ROOT / args.captures_dir / run_tag
                validation_summary = load_json(capture_dir / "mqam_validation.json")
                experiment_summary = load_json(capture_dir / "receiver_experiment.json")
                best_phase = validation_summary["best_phase"]
                lock_score = best_phase["lock_score"]
                good_lock = bool(lock_score >= args.lock_threshold)
                attempt_row = {
                    "pilot_symbols": pilot_symbols,
                    "repetition": repetition + 1,
                    "attempt": attempt,
                    "selected": False,
                    "total_symbols": args.total_symbols,
                    "payload_symbols": args.total_symbols - pilot_symbols,
                    "capture_dir": str(capture_dir),
                    "lock_score": lock_score,
                    "good_lock": good_lock,
                    "included": good_lock if args.exclude_bad_lock else True,
                    "validation_ber": best_phase["overall_ber"],
                    "validation_mean_pilot_ber": best_phase["mean_pilot_ber"],
                    "validation_mean_snr_db": best_phase.get("mean_snr_db"),
                }
                for detector_name, detector_result in experiment_summary["detectors"].items():
                    attempt_row[f"{detector_name}_ber"] = detector_result["overall_ber"]
                    attempt_row[f"{detector_name}_ser"] = detector_result.get("overall_ser", detector_result["overall_ber"])
                attempt_rows.append(attempt_row)

                if good_lock:
                    selected_row = dict(attempt_row)
                    selected_row["selected"] = True
                    print(f"Accepted lock_score={lock_score:.4f} on attempt {attempt}.")
                    break

                print(f"Rejected lock_score={lock_score:.4f} below threshold {args.lock_threshold:.4f}.")
                if not args.retry_until_lock:
                    selected_row = dict(attempt_row)
                    selected_row["selected"] = True
                    break
                if args.max_lock_attempts > 0 and attempt >= args.max_lock_attempts:
                    selected_row = dict(attempt_row)
                    selected_row["selected"] = True
                    print(
                        f"Max attempts reached for k={pilot_symbols}, rep={repetition + 1}; "
                        "using final bad-lock attempt."
                    )
                    break
                if args.retry_delay > 0:
                    print(f"Retrying in {args.retry_delay:.1f} s...")
                    time.sleep(args.retry_delay)
                attempt += 1

            if selected_row is None:
                raise RuntimeError(f"No selected run for k={pilot_symbols}, repetition={repetition + 1}.")
            for row in reversed(attempt_rows):
                if row["capture_dir"] == selected_row["capture_dir"]:
                    row["selected"] = True
                    break
            repetition_rows.append(selected_row)

            if args.inter_run_delay > 0 and repetition + 1 < args.repetitions:
                print(f"Settling for {args.inter_run_delay:.1f} s before next repetition...")
                time.sleep(args.inter_run_delay)

        included_rows = [row for row in repetition_rows if row["included"]]
        good_lock_rows = [row for row in repetition_rows if row["good_lock"]]
        selected_attempts = [row["attempt"] for row in repetition_rows]
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
                "total_attempts": int(sum(1 for row in attempt_rows if row["pilot_symbols"] == pilot_symbols)),
                "attempts_per_selected_run": summarize_metric(selected_attempts),
                "sync_success_rate": float(sum(1 for row in repetition_rows if row["good_lock"]) / len(repetition_rows)),
                "runs": repetition_rows,
                "attempts": [row for row in attempt_rows if row["pilot_symbols"] == pilot_symbols],
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
        "remote_host": remote_host,
        "remote_role": args.remote_role,
        "remote_os": args.remote_os,
        "remote_repo": args.remote_repo,
        "remote_conda_env": args.remote_conda_env,
        "remote_conda_exe": args.remote_conda_exe,
        "modulation": args.modulation,
        "total_symbols": args.total_symbols,
        "pilot_symbols_tested": args.pilot_symbols,
        "detectors": args.detectors,
        "repetitions": args.repetitions,
        "lock_threshold": args.lock_threshold,
        "retry_until_lock": args.retry_until_lock,
        "max_lock_attempts": args.max_lock_attempts,
        "retry_delay": args.retry_delay,
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
        "total_attempts",
        "attempts_per_selected_run_mean",
        "attempts_per_selected_run_max",
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
                "total_attempts": row["total_attempts"],
                "attempts_per_selected_run_mean": row["attempts_per_selected_run"]["mean"],
                "attempts_per_selected_run_max": row["attempts_per_selected_run"]["max"],
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
        "attempt",
        "selected",
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
