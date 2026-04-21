#!/usr/bin/env python3
"""
Automated square M-QAM transmit/receive/validate test.

This is the generic companion to the working QPSK harness. It uses the new
headless M-QAM TX/RX scripts and validates against the processed oversampled
stream with the shared post-recovery validator.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from core.modulation import normalize_modulation


def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(str(path))


def stream_subprocess_output(label: str, proc: subprocess.Popen[str]) -> None:
    if proc.stdout is None:
        return
    for line in proc.stdout:
        sys.stdout.write(f"[{label}] {line}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fresh headless M-QAM SDR test and validate BER.")
    parser.add_argument("--modulation", required=True, choices=["QPSK", "16QAM", "64QAM"])
    parser.add_argument("--tx-frame", default=None)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--rx-time", type=float, default=6.0)
    parser.add_argument("--tx-delay", type=float, default=1.5)
    parser.add_argument("--tx-time", type=float, default=None)
    parser.add_argument("--captures-dir", default="captures")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--validator-prefix", default="mqam_validation")
    parser.add_argument("--rx-device-args", default="driver=plutosdr")
    parser.add_argument("--tx-device-args", default="driver=plutosdr")
    parser.add_argument("--rx-gain", type=float, default=40.0)
    parser.add_argument("--carrier-recovery", choices=["none", "costas"], default="none")
    parser.add_argument("--costas-bw", type=float, default=0.001)
    parser.add_argument("--tx-gain", type=float, default=50.0)
    parser.add_argument("--tx-scale", type=float, default=0.2)
    parser.add_argument("--run-experiment", action="store_true")
    parser.add_argument("--experiment-detectors", nargs="+", default=["mmse", "mmse_df"])
    parser.add_argument("--experiment-output", default="receiver_experiment.json")
    parser.add_argument("--icl-checkpoint", default=None)
    parser.add_argument("--defined-checkpoint", default=None)
    return parser.parse_args()


def preflight_python(python_exe: str, root: Path) -> None:
    checks = [
        ("GNU Radio", "import gnuradio"),
        ("GNU Radio Soapy", "from gnuradio import soapy"),
        ("SoapySDR", "import SoapySDR"),
    ]
    for label, code in checks:
        result = subprocess.run([python_exe, "-c", code], cwd=root, capture_output=True, text=True)
        if result.returncode != 0:
            stderr = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"{label} preflight failed for interpreter {python_exe}\n{stderr}")


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    modulation = normalize_modulation(args.modulation)
    modulation_tag = modulation.lower()
    tx_frame = root / (args.tx_frame or f"frames/test_frame_{modulation_tag}.bin")
    metadata = root / (args.metadata or f"frames/test_frame_{modulation_tag}.json")

    preflight_python(args.python, root)
    ensure_exists(tx_frame)
    ensure_exists(metadata)

    if modulation in {"16QAM", "64QAM"} and args.carrier_recovery == "costas":
        raise ValueError(
            f"--carrier-recovery costas is not supported for {modulation} in "
            "Receive_Signal_MQAM_Headless.py. Use --carrier-recovery none."
        )

    if args.tx_delay >= args.rx_time:
        raise ValueError("--tx-delay must be smaller than --rx-time")

    tx_time = args.tx_time if args.tx_time is not None else max(2.5, args.rx_time - args.tx_delay - 1.0)
    tag = args.tag or time.strftime(f"{modulation_tag}_headless_%Y%m%d_%H%M%S")
    capture_dir = (root / args.captures_dir / tag).resolve()
    capture_dir.mkdir(parents=True, exist_ok=False)

    rx_processed = capture_dir / f"received_processed_{modulation_tag}.bin"
    rx_raw = capture_dir / f"received_iq_raw_{modulation_tag}.bin"
    tx_frame_copy = capture_dir / "transmitted_frame.bin"
    metadata_copy = capture_dir / "frame_metadata.json"
    validator_prefix = capture_dir / args.validator_prefix
    runner_log = capture_dir / "run_metadata.json"

    shutil.copy2(tx_frame, tx_frame_copy)
    frame_meta = json.loads(metadata.read_text(encoding="utf-8"))

    copied_payload_files = []
    for payload_path in frame_meta.get("payload_files", []):
        src = root / payload_path
        if src.exists():
            dst = capture_dir / payload_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied_payload_files.append(str(dst))
        else:
            copied_payload_files.append(payload_path)
    frame_meta["payload_files"] = copied_payload_files
    metadata_copy.write_text(json.dumps(frame_meta, indent=2), encoding="utf-8")

    rx_cmd = [
        args.python,
        str(root / "Receive_Signal_MQAM_Headless.py"),
        str(args.rx_time),
        str(rx_processed),
        str(rx_raw),
        args.rx_device_args,
        modulation,
        str(args.rx_gain),
        args.carrier_recovery,
        str(args.costas_bw),
    ]
    tx_cmd = [
        args.python,
        str(root / "Send_Signal_MQAM_Headless.py"),
        str(tx_frame_copy),
        modulation,
        str(tx_time),
        "true",
        args.tx_device_args,
        str(args.tx_gain),
        str(args.tx_scale),
    ]

    print("=" * 72)
    print(f"{modulation} HEADLESS SDR TEST")
    print("=" * 72)
    print(f"Capture dir:       {capture_dir}")
    print(f"RX runtime:        {args.rx_time:.1f} s")
    print(f"TX delay:          {args.tx_delay:.1f} s")
    print(f"TX runtime:        {tx_time:.1f} s")
    print(f"TX frame:          {tx_frame_copy}")
    print(f"RX device:         {args.rx_device_args}")
    print(f"TX device:         {args.tx_device_args}")
    print(f"RX gain:           {args.rx_gain}")
    print(f"Carrier recovery:  {args.carrier_recovery}")
    print(f"Costas BW:         {args.costas_bw}")
    print(f"TX gain:           {args.tx_gain}")
    print(f"TX scale:          {args.tx_scale}")
    print(f"Experiment:        {args.run_experiment}")
    print()

    receiver = subprocess.Popen(
        rx_cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    print(f"Receiver PID: {receiver.pid}")
    time.sleep(args.tx_delay)
    transmitter = subprocess.Popen(
        tx_cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    print(f"Transmitter PID: {transmitter.pid}")
    print()

    stream_subprocess_output("RX", receiver)
    stream_subprocess_output("TX", transmitter)

    rx_code = receiver.wait()
    tx_code = transmitter.wait()
    if rx_code != 0:
        raise RuntimeError(f"Receiver exited with code {rx_code}")
    if tx_code != 0:
        raise RuntimeError(f"Transmitter exited with code {tx_code}")

    validate_cmd = [
        args.python,
        str(root / "qpsk_validate_from_post_costas.py"),
        "--capture-dir",
        str(capture_dir),
        "--post-costas-file",
        rx_processed.name,
        "--tx-frame",
        tx_frame_copy.name,
        "--metadata",
        metadata_copy.name,
        "--output-prefix",
        args.validator_prefix,
    ]

    print()
    print("Running pilot-based validation...")
    validation = subprocess.run(validate_cmd, cwd=root, text=True, capture_output=True)
    sys.stdout.write(validation.stdout)
    if validation.stderr:
        sys.stderr.write(validation.stderr)
    if validation.returncode != 0:
        raise RuntimeError(f"Validator exited with code {validation.returncode}")

    summary_path = validator_prefix.with_suffix(".json")
    ensure_exists(summary_path)

    experiment_summary_path = None
    if args.run_experiment:
        experiment_cmd = [
            args.python,
            str(root / "receiver_experiment.py"),
            "--capture-dir",
            str(capture_dir),
            "--post-costas-file",
            rx_processed.name,
            "--tx-frame",
            tx_frame_copy.name,
            "--metadata",
            metadata_copy.name,
            "--output",
            args.experiment_output,
            "--detectors",
            *args.experiment_detectors,
        ]
        if args.icl_checkpoint:
            experiment_cmd.extend(["--icl-checkpoint", args.icl_checkpoint])
        if args.defined_checkpoint:
            experiment_cmd.extend(["--defined-checkpoint", args.defined_checkpoint])
        print()
        print("Running modular detector experiment...")
        experiment = subprocess.run(experiment_cmd, cwd=root, text=True, capture_output=True)
        sys.stdout.write(experiment.stdout)
        if experiment.stderr:
            sys.stderr.write(experiment.stderr)
        if experiment.returncode != 0:
            raise RuntimeError(f"Receiver experiment exited with code {experiment.returncode}")
        experiment_summary_path = str(capture_dir / args.experiment_output)

    run_summary = {
        "capture_dir": str(capture_dir),
        "modulation": modulation,
        "tx_frame": str(tx_frame_copy),
        "metadata": str(metadata_copy),
        "rx_processed": str(rx_processed),
        "rx_raw_iq": str(rx_raw),
        "rx_time_s": args.rx_time,
        "tx_delay_s": args.tx_delay,
        "tx_time_s": tx_time,
        "rx_gain": args.rx_gain,
        "carrier_recovery": args.carrier_recovery,
        "costas_bw": args.costas_bw,
        "tx_gain": args.tx_gain,
        "tx_scale": args.tx_scale,
        "run_experiment": args.run_experiment,
        "experiment_detectors": args.experiment_detectors,
        "validator_summary": str(summary_path),
        "experiment_summary": experiment_summary_path,
    }
    runner_log.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    print()
    print(f"Run metadata: {runner_log}")
    print(f"Validation summary: {summary_path}")
    if experiment_summary_path:
        print(f"Experiment summary: {experiment_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
