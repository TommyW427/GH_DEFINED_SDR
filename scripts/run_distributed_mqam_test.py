#!/usr/bin/env python3
"""
Run one two-computer PlutoSDR transmit/receive/validate experiment.

This script is intended to be launched on the TX/controller computer. It starts
the receiver on a remote computer over SSH, transmits locally after a short
delay, copies the remote capture files back with SCP, and then runs the normal
local validator and detector experiment.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SRC_DIR))

from core.modulation import normalize_modulation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a distributed M-QAM SDR test over SSH.")
    parser.add_argument("--modulation", required=True, choices=["BPSK", "QPSK", "16QAM", "64QAM"])
    parser.add_argument("--tx-frame", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--remote-rx-host", required=True, help="SSH target for the RX computer, e.g. user@rx-host")
    parser.add_argument(
        "--remote-repo",
        default=str(REPO_ROOT),
        help="Path to this repository on the RX computer. Default assumes identical path on both computers.",
    )
    parser.add_argument("--remote-python", default=sys.executable)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--ssh-port", type=int, default=None)
    parser.add_argument("--ssh-option", action="append", default=[], help="Extra SSH/SCP -o option; may be repeated.")
    parser.add_argument("--rx-time", type=float, default=6.0)
    parser.add_argument("--tx-delay", type=float, default=1.5)
    parser.add_argument("--tx-time", type=float, default=None)
    parser.add_argument("--captures-dir", default="captures")
    parser.add_argument("--tag", default=None)
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
    parser.add_argument("--keep-remote-capture", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print SSH/TX/SCP commands without running them.")
    return parser.parse_args()


def local_env() -> dict[str, str]:
    env = os.environ.copy()
    py_path = f"{SRC_DIR}{os.pathsep}{SCRIPTS_DIR}"
    env["PYTHONPATH"] = py_path if not env.get("PYTHONPATH") else f"{py_path}{os.pathsep}{env['PYTHONPATH']}"
    return env


def ssh_base(args: argparse.Namespace) -> list[str]:
    cmd = ["ssh"]
    if args.ssh_port is not None:
        cmd.extend(["-p", str(args.ssh_port)])
    for option in args.ssh_option:
        cmd.extend(["-o", option])
    cmd.append(args.remote_rx_host)
    return cmd


def scp_base(args: argparse.Namespace) -> list[str]:
    cmd = ["scp"]
    if args.ssh_port is not None:
        cmd.extend(["-P", str(args.ssh_port)])
    for option in args.ssh_option:
        cmd.extend(["-o", option])
    return cmd


def remote_path(remote_repo: str, *parts: str) -> str:
    return "/".join([remote_repo.rstrip("/"), *[part.strip("/") for part in parts]])


def remote_shell_command(args: argparse.Namespace, remote_capture: str, remote_processed: str, remote_raw: str) -> str:
    repo = args.remote_repo.rstrip("/")
    rx_script = remote_path(repo, "scripts", "Receive_Signal_MQAM_Headless.py")
    remote_pythonpath = f"{remote_path(repo, 'src')}:{remote_path(repo, 'scripts')}"
    rx_cmd = [
        "env",
        f"PYTHONPATH={remote_pythonpath}",
        args.remote_python,
        rx_script,
        str(args.rx_time),
        remote_processed,
        remote_raw,
        args.rx_device_args,
        normalize_modulation(args.modulation),
        str(args.rx_gain),
        args.carrier_recovery,
        str(args.costas_bw),
    ]
    return " && ".join(
        [
            f"mkdir -p {shlex.quote(remote_capture)}",
            f"cd {shlex.quote(repo)}",
            shlex.join(rx_cmd),
        ]
    )


def run_checked(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)
    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}: {shlex.join(cmd)}")
    return result


def copy_payloads_into_capture(metadata_path: Path, capture_dir: Path, metadata_copy: Path, tx_bits_root: Path) -> None:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    copied_payload_files = []
    for payload_file in metadata.get("payload_files", []):
        src = Path(payload_file)
        if not src.is_absolute():
            src = tx_bits_root / src
        if src.exists():
            dst = capture_dir / Path(payload_file).name
            shutil.copy2(src, dst)
            copied_payload_files.append(dst.name)
        else:
            copied_payload_files.append(payload_file)
    metadata["payload_files"] = copied_payload_files
    metadata_copy.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    modulation = normalize_modulation(args.modulation)
    modulation_tag = modulation.lower()
    if modulation in {"16QAM", "64QAM"} and args.carrier_recovery == "costas":
        raise ValueError(f"--carrier-recovery costas is not supported for {modulation}. Use --carrier-recovery none.")
    if args.tx_delay >= args.rx_time:
        raise ValueError("--tx-delay must be smaller than --rx-time")

    tx_frame = (REPO_ROOT / args.tx_frame).resolve() if not Path(args.tx_frame).is_absolute() else Path(args.tx_frame)
    metadata = (REPO_ROOT / args.metadata).resolve() if not Path(args.metadata).is_absolute() else Path(args.metadata)
    if not tx_frame.exists():
        raise FileNotFoundError(str(tx_frame))
    if not metadata.exists():
        raise FileNotFoundError(str(metadata))

    tag = args.tag or time.strftime(f"{modulation_tag}_distributed_%Y%m%d_%H%M%S")
    capture_dir = (REPO_ROOT / args.captures_dir / tag).resolve()
    capture_dir.mkdir(parents=True, exist_ok=False)

    tx_time = args.tx_time if args.tx_time is not None else max(2.5, args.rx_time - args.tx_delay - 1.0)
    tx_frame_copy = capture_dir / "transmitted_frame.bin"
    metadata_copy = capture_dir / "frame_metadata.json"
    local_processed = capture_dir / f"received_processed_{modulation_tag}.bin"
    local_raw = capture_dir / f"received_iq_raw_{modulation_tag}.bin"
    remote_capture = remote_path(args.remote_repo, args.captures_dir, tag)
    remote_processed = remote_path(remote_capture, local_processed.name)
    remote_raw = remote_path(remote_capture, local_raw.name)

    shutil.copy2(tx_frame, tx_frame_copy)
    copy_payloads_into_capture(metadata, capture_dir, metadata_copy, REPO_ROOT)

    remote_cmd = remote_shell_command(args, remote_capture, remote_processed, remote_raw)
    ssh_cmd = [*ssh_base(args), remote_cmd]
    tx_cmd = [
        args.python,
        str(SCRIPTS_DIR / "Send_Signal_MQAM_Headless.py"),
        str(tx_frame_copy),
        modulation,
        str(tx_time),
        "true",
        args.tx_device_args,
        str(args.tx_gain),
        str(args.tx_scale),
    ]
    scp_processed = [*scp_base(args), f"{args.remote_rx_host}:{shlex.quote(remote_processed)}", str(local_processed)]
    scp_raw = [*scp_base(args), f"{args.remote_rx_host}:{shlex.quote(remote_raw)}", str(local_raw)]

    print("=" * 72)
    print(f"{modulation} DISTRIBUTED SDR TEST")
    print("=" * 72)
    print(f"Local capture dir: {capture_dir}")
    print(f"Remote RX host:    {args.remote_rx_host}")
    print(f"Remote repo:       {args.remote_repo}")
    print(f"Remote capture:    {remote_capture}")
    print(f"RX runtime:        {args.rx_time:.1f} s")
    print(f"TX delay:          {args.tx_delay:.1f} s")
    print(f"TX runtime:        {tx_time:.1f} s")
    print(f"RX device:         {args.rx_device_args}")
    print(f"TX device:         {args.tx_device_args}")
    print(f"Carrier recovery:  {args.carrier_recovery}")
    print()

    if args.dry_run:
        print("SSH RX command:")
        print(shlex.join(ssh_cmd))
        print("\nLocal TX command:")
        print(shlex.join(tx_cmd))
        print("\nSCP commands:")
        print(shlex.join(scp_processed))
        print(shlex.join(scp_raw))
        return 0

    receiver = subprocess.Popen(ssh_cmd, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(f"Remote receiver SSH PID: {receiver.pid}")
    time.sleep(args.tx_delay)

    print("Starting local transmitter...")
    tx_result = subprocess.run(tx_cmd, cwd=REPO_ROOT, env=local_env(), text=True, capture_output=True)
    if tx_result.stdout:
        sys.stdout.write(tx_result.stdout)
    if tx_result.stderr:
        sys.stderr.write(tx_result.stderr)

    rx_stdout, _ = receiver.communicate()
    if rx_stdout:
        for line in rx_stdout.splitlines():
            print(f"[REMOTE_RX] {line}")
    if receiver.returncode != 0:
        raise RuntimeError(f"Remote receiver exited with code {receiver.returncode}")
    if tx_result.returncode != 0:
        raise RuntimeError(f"Local transmitter exited with code {tx_result.returncode}")

    print("Copying remote receiver artifacts back to controller...")
    run_checked(scp_processed, cwd=REPO_ROOT)
    run_checked(scp_raw, cwd=REPO_ROOT)

    validate_cmd = [
        args.python,
        str(SCRIPTS_DIR / "qpsk_validate_from_post_costas.py"),
        "--capture-dir",
        str(capture_dir),
        "--post-costas-file",
        local_processed.name,
        "--tx-frame",
        tx_frame_copy.name,
        "--metadata",
        metadata_copy.name,
        "--output-prefix",
        args.validator_prefix,
    ]
    print("\nRunning pilot-based validation...")
    run_checked(validate_cmd, cwd=REPO_ROOT, env=local_env())

    experiment_summary_path = None
    if args.run_experiment:
        experiment_cmd = [
            args.python,
            str(SCRIPTS_DIR / "receiver_experiment.py"),
            "--capture-dir",
            str(capture_dir),
            "--post-costas-file",
            local_processed.name,
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
        print("\nRunning modular detector experiment...")
        run_checked(experiment_cmd, cwd=REPO_ROOT, env=local_env())
        experiment_summary_path = str(capture_dir / args.experiment_output)

    if not args.keep_remote_capture:
        cleanup_cmd = [*ssh_base(args), f"rm -rf {shlex.quote(remote_capture)}"]
        run_checked(cleanup_cmd, cwd=REPO_ROOT)

    run_summary = {
        "capture_dir": str(capture_dir),
        "remote_rx_host": args.remote_rx_host,
        "remote_repo": args.remote_repo,
        "remote_capture": remote_capture,
        "modulation": modulation,
        "tx_frame": str(tx_frame_copy),
        "metadata": str(metadata_copy),
        "rx_processed": str(local_processed),
        "rx_raw_iq": str(local_raw),
        "rx_time_s": args.rx_time,
        "tx_delay_s": args.tx_delay,
        "tx_time_s": tx_time,
        "rx_gain": args.rx_gain,
        "carrier_recovery": args.carrier_recovery,
        "costas_bw": args.costas_bw,
        "tx_gain": args.tx_gain,
        "tx_scale": args.tx_scale,
        "validator_summary": str(capture_dir / f"{args.validator_prefix}.json"),
        "experiment_summary": experiment_summary_path,
    }
    run_metadata = capture_dir / "distributed_run_metadata.json"
    run_metadata.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    print()
    print(f"Distributed run metadata: {run_metadata}")
    print(f"Validation summary:       {capture_dir / f'{args.validator_prefix}.json'}")
    if experiment_summary_path:
        print(f"Experiment summary:       {experiment_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
