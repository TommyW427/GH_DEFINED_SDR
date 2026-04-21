#!/usr/bin/env python3
"""
Run one two-computer PlutoSDR transmit/receive/validate experiment.

Default topology:
  - launch this script on the RX/controller computer
  - start the local receiver
  - SSH to the remote TX computer and transmit from its PlutoSDR
  - validate the local RX capture

The legacy inverse topology is still available with --remote-role rx.
"""

from __future__ import annotations

import argparse
import base64
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
    parser.add_argument("--remote-host", default=None, help="SSH target for the remote computer.")
    parser.add_argument("--remote-tx-host", default=None, help="Alias for --remote-host when --remote-role tx.")
    parser.add_argument("--remote-rx-host", default=None, help="Alias for --remote-host when --remote-role rx.")
    parser.add_argument("--remote-role", choices=["tx", "rx"], default="tx")
    parser.add_argument(
        "--remote-os",
        choices=["windows", "posix"],
        default="windows",
        help="Remote shell style. Use windows for OpenSSH into PowerShell/cmd.",
    )
    parser.add_argument(
        "--remote-repo",
        default="C:/Users/tpdub/Downloads/Wireless/GH_DEFINED_SDR",
        help="Path to this repository on the remote computer.",
    )
    parser.add_argument("--remote-python", default="python")
    parser.add_argument(
        "--remote-conda-env",
        default="gnuradio",
        help="Optional conda environment to activate on the remote computer before running TX/RX.",
    )
    parser.add_argument(
        "--remote-conda-exe",
        default="conda",
        help="Remote conda executable or full path. Used only with --remote-conda-env.",
    )
    parser.add_argument(
        "--remote-sdr-preflight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Check remote SoapySDR modules/devices before launching the remote radio process.",
    )
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
    parser.add_argument("--experiment-detectors", nargs="+", default=["mmse", "mmse_df", "icl", "defined"])
    parser.add_argument("--experiment-output", default="receiver_experiment.json")
    parser.add_argument("--icl-checkpoint", default=None)
    parser.add_argument("--defined-checkpoint", default=None)
    parser.add_argument("--keep-remote-capture", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print SSH/TX/RX/SCP commands without running them.")
    return parser.parse_args()


def resolve_remote_host(args: argparse.Namespace) -> str:
    host = args.remote_host
    if args.remote_role == "tx":
        host = host or args.remote_tx_host
    else:
        host = host or args.remote_rx_host
    if not host:
        raise ValueError("Provide --remote-host, --remote-tx-host, or --remote-rx-host.")
    return host


def local_env() -> dict[str, str]:
    env = os.environ.copy()
    py_path = f"{SRC_DIR}{os.pathsep}{SCRIPTS_DIR}"
    env["PYTHONPATH"] = py_path if not env.get("PYTHONPATH") else f"{py_path}{os.pathsep}{env['PYTHONPATH']}"
    return env


def ssh_base(args: argparse.Namespace, remote_host: str) -> list[str]:
    cmd = ["ssh"]
    if args.ssh_port is not None:
        cmd.extend(["-p", str(args.ssh_port)])
    for option in args.ssh_option:
        cmd.extend(["-o", option])
    cmd.append(remote_host)
    return cmd


def scp_base(args: argparse.Namespace) -> list[str]:
    cmd = ["scp"]
    if args.ssh_port is not None:
        cmd.extend(["-P", str(args.ssh_port)])
    for option in args.ssh_option:
        cmd.extend(["-o", option])
    return cmd


def normalize_remote_path(path: str, remote_os: str) -> str:
    if remote_os == "windows":
        return path.replace("\\", "/")
    return path


def remote_path(remote_os: str, remote_repo: str, *parts: str) -> str:
    repo = normalize_remote_path(remote_repo, remote_os).rstrip("/\\")
    sep = "/" if remote_os == "windows" else "/"
    return sep.join([repo, *[part.strip("/\\") for part in parts]])


def ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def win_quote(value: str) -> str:
    return '"' + value.replace('"', r'\"') + '"'


def base64_encode(value: str) -> str:
    return base64.b64encode(value.encode("utf-8")).decode("ascii")


def remote_command(args: argparse.Namespace, argv: list[str], remote_cwd: str, remote_capture: str) -> str:
    remote_cwd = normalize_remote_path(remote_cwd, args.remote_os)
    remote_capture = normalize_remote_path(remote_capture, args.remote_os)
    if args.remote_os == "windows":
        pythonpath = f"{remote_path(args.remote_os, args.remote_repo, 'src')};{remote_path(args.remote_os, args.remote_repo, 'scripts')}"
        argv_json = json.dumps(argv)
        argv_json_b64 = base64_encode(argv_json)
        helper = remote_path(args.remote_os, args.remote_repo, "scripts", "windows_remote_run.ps1")
        return " ".join(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                win_quote(helper),
                "-Repo",
                win_quote(remote_cwd),
                "-Capture",
                win_quote(remote_capture),
                "-CondaExe",
                win_quote(args.remote_conda_exe),
                "-CondaEnv",
                win_quote(args.remote_conda_env or ""),
                "-PythonPath",
                win_quote(pythonpath),
                "-ArgvJsonBase64",
                win_quote(argv_json_b64),
            ]
        )

    pythonpath = f"{remote_path(args.remote_os, args.remote_repo, 'src')}:{remote_path(args.remote_os, args.remote_repo, 'scripts')}"
    parts = [
        f"mkdir -p {shlex.quote(remote_capture)}",
        f"cd {shlex.quote(remote_cwd)}",
    ]
    if args.remote_conda_env:
        parts.extend(
            [
                f"eval \"$({shlex.quote(args.remote_conda_exe)} shell.posix hook)\"",
                f"conda activate {shlex.quote(args.remote_conda_env)}",
            ]
        )
    parts.append(" ".join(["env", shlex.quote(f"PYTHONPATH={pythonpath}"), *[shlex.quote(item) for item in argv]]))
    return " && ".join(parts)


def windows_simple_command(command: str) -> str:
    encoded = base64.b64encode(command.encode("utf-16le")).decode("ascii")
    return f"powershell.exe -NoProfile -ExecutionPolicy Bypass -EncodedCommand {encoded}"


def remote_cleanup_command(args: argparse.Namespace, remote_capture: str) -> str:
    remote_capture = normalize_remote_path(remote_capture, args.remote_os)
    if args.remote_os == "windows":
        ps = f"Remove-Item -LiteralPath {ps_quote(remote_capture)} -Recurse -Force -ErrorAction SilentlyContinue"
        encoded = base64.b64encode(ps.encode("utf-16le")).decode("ascii")
        return f"powershell.exe -NoProfile -ExecutionPolicy Bypass -EncodedCommand {encoded}"
    return f"rm -rf {shlex.quote(remote_capture)}"


def scp_remote_spec(remote_host: str, remote_file: str, remote_os: str) -> str:
    return f"{remote_host}:{normalize_remote_path(remote_file, remote_os)}"


def run_checked(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)
    if result.stdout:
        sys.stdout.write(result.stdout)
    write_stderr(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}: {shlex.join(cmd)}")
    return result


def clean_powershell_clixml(text: str) -> str:
    if not text:
        return ""
    if "#< CLIXML" not in text and "<Objs Version=" not in text:
        return text
    if 'S="Error"' in text:
        return text
    lines = [
        line
        for line in text.splitlines()
        if "#< CLIXML" not in line
        and "<Objs Version=" not in line
        and "Preparing modules for first use." not in line
    ]
    return "\n".join(lines).strip()


def write_stderr(text: str | None) -> None:
    cleaned = clean_powershell_clixml(text or "")
    if cleaned:
        sys.stderr.write(cleaned)
        if not cleaned.endswith("\n"):
            sys.stderr.write("\n")


def remote_soapy_preflight_argv(args: argparse.Namespace, device_args: str, direction: str) -> list[str]:
    return [
        args.remote_python,
        remote_path(args.remote_os, args.remote_repo, "scripts", "remote_soapy_preflight.py"),
        "--device-args",
        device_args,
        "--direction",
        direction,
    ]


def run_remote_preflight(
    args: argparse.Namespace,
    remote_host: str,
    remote_capture: str,
    device_args: str,
    direction: str,
) -> None:
    if not args.remote_sdr_preflight:
        return
    print(f"Running remote {direction} SoapySDR preflight...")
    argv = remote_soapy_preflight_argv(args, device_args, direction)
    cmd = [*ssh_base(args, remote_host), remote_command(args, argv, args.remote_repo, remote_capture)]
    run_checked(cmd, cwd=REPO_ROOT)


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


def run_validation_and_experiment(
    args: argparse.Namespace,
    capture_dir: Path,
    local_processed: Path,
    tx_frame_copy: Path,
    metadata_copy: Path,
) -> str | None:
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

    if not args.run_experiment:
        return None

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
    return str(capture_dir / args.experiment_output)


def main() -> int:
    args = parse_args()
    remote_host = resolve_remote_host(args)
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
    remote_capture = remote_path(args.remote_os, args.remote_repo, args.captures_dir, tag)
    remote_tx_frame = remote_path(args.remote_os, remote_capture, "transmitted_frame.bin")
    remote_processed = remote_path(args.remote_os, remote_capture, local_processed.name)
    remote_raw = remote_path(args.remote_os, remote_capture, local_raw.name)

    shutil.copy2(tx_frame, tx_frame_copy)
    copy_payloads_into_capture(metadata, capture_dir, metadata_copy, REPO_ROOT)

    print("=" * 72)
    print(f"{modulation} DISTRIBUTED SDR TEST")
    print("=" * 72)
    print(f"Topology:          remote {args.remote_role.upper()}, local {'RX' if args.remote_role == 'tx' else 'TX'}")
    print(f"Local capture dir: {capture_dir}")
    print(f"Remote host:       {remote_host}")
    print(f"Remote OS:         {args.remote_os}")
    print(f"Remote repo:       {args.remote_repo}")
    if args.remote_conda_env:
        print(f"Remote conda env:  {args.remote_conda_env}")
    print(f"Remote capture:    {remote_capture}")
    print(f"RX runtime:        {args.rx_time:.1f} s")
    print(f"TX delay:          {args.tx_delay:.1f} s")
    print(f"TX runtime:        {tx_time:.1f} s")
    print(f"RX device:         {args.rx_device_args}")
    print(f"TX device:         {args.tx_device_args}")
    print(f"Carrier recovery:  {args.carrier_recovery}")
    print(f"Detectors:         {' '.join(args.experiment_detectors)}")
    print()

    remote_mkdir = remote_command(args, [args.remote_python, "-c", "print('remote ready')"], args.remote_repo, remote_capture)

    if args.remote_role == "tx":
        rx_cmd = [
            args.python,
            str(SCRIPTS_DIR / "Receive_Signal_MQAM_Headless.py"),
            str(args.rx_time),
            str(local_processed),
            str(local_raw),
            args.rx_device_args,
            modulation,
            str(args.rx_gain),
            args.carrier_recovery,
            str(args.costas_bw),
        ]
        remote_tx_argv = [
            args.remote_python,
            remote_path(args.remote_os, args.remote_repo, "scripts", "Send_Signal_MQAM_Headless.py"),
            remote_tx_frame,
            modulation,
            str(tx_time),
            "true",
            args.tx_device_args,
            str(args.tx_gain),
            str(args.tx_scale),
        ]
        ssh_tx_cmd = [*ssh_base(args, remote_host), remote_command(args, remote_tx_argv, args.remote_repo, remote_capture)]
        scp_frame_cmd = [*scp_base(args), str(tx_frame_copy), scp_remote_spec(remote_host, remote_tx_frame, args.remote_os)]

        if args.dry_run:
            print("Remote setup command:")
            print(shlex.join([*ssh_base(args, remote_host), remote_mkdir]))
            if args.remote_sdr_preflight:
                print("\nRemote TX SoapySDR preflight command:")
                preflight_cmd = [
                    *ssh_base(args, remote_host),
                    remote_command(
                        args,
                        remote_soapy_preflight_argv(args, args.tx_device_args, "TX"),
                        args.remote_repo,
                        remote_capture,
                    ),
                ]
                print(shlex.join(preflight_cmd))
            print("\nSCP frame-to-TX command:")
            print(shlex.join(scp_frame_cmd))
            print("\nLocal RX command:")
            print(shlex.join(rx_cmd))
            print("\nRemote TX command:")
            print(shlex.join(ssh_tx_cmd))
            return 0

        run_checked([*ssh_base(args, remote_host), remote_mkdir], cwd=REPO_ROOT)
        run_remote_preflight(args, remote_host, remote_capture, args.tx_device_args, "TX")
        run_checked(scp_frame_cmd, cwd=REPO_ROOT)

        receiver = subprocess.Popen(rx_cmd, cwd=REPO_ROOT, env=local_env(), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(f"Local receiver PID: {receiver.pid}")
        time.sleep(args.tx_delay)

        print("Starting remote transmitter...")
        tx_result = subprocess.run(ssh_tx_cmd, cwd=REPO_ROOT, text=True, capture_output=True)
        if tx_result.stdout:
            for line in tx_result.stdout.splitlines():
                print(f"[REMOTE_TX] {line}")
        write_stderr(tx_result.stderr)

        rx_stdout, _ = receiver.communicate()
        if rx_stdout:
            for line in rx_stdout.splitlines():
                print(f"[LOCAL_RX] {line}")
        if receiver.returncode != 0:
            raise RuntimeError(f"Local receiver exited with code {receiver.returncode}")
        if tx_result.returncode != 0:
            raise RuntimeError(f"Remote transmitter exited with code {tx_result.returncode}")

    else:
        remote_rx_argv = [
            args.remote_python,
            remote_path(args.remote_os, args.remote_repo, "scripts", "Receive_Signal_MQAM_Headless.py"),
            str(args.rx_time),
            remote_processed,
            remote_raw,
            args.rx_device_args,
            modulation,
            str(args.rx_gain),
            args.carrier_recovery,
            str(args.costas_bw),
        ]
        ssh_rx_cmd = [*ssh_base(args, remote_host), remote_command(args, remote_rx_argv, args.remote_repo, remote_capture)]
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
        scp_processed = [*scp_base(args), scp_remote_spec(remote_host, remote_processed, args.remote_os), str(local_processed)]
        scp_raw = [*scp_base(args), scp_remote_spec(remote_host, remote_raw, args.remote_os), str(local_raw)]

        if args.dry_run:
            print("Remote RX command:")
            print(shlex.join(ssh_rx_cmd))
            if args.remote_sdr_preflight:
                print("\nRemote RX SoapySDR preflight command:")
                preflight_cmd = [
                    *ssh_base(args, remote_host),
                    remote_command(
                        args,
                        remote_soapy_preflight_argv(args, args.rx_device_args, "RX"),
                        args.remote_repo,
                        remote_capture,
                    ),
                ]
                print(shlex.join(preflight_cmd))
            print("\nLocal TX command:")
            print(shlex.join(tx_cmd))
            print("\nSCP commands:")
            print(shlex.join(scp_processed))
            print(shlex.join(scp_raw))
            return 0

        run_remote_preflight(args, remote_host, remote_capture, args.rx_device_args, "RX")
        receiver = subprocess.Popen(ssh_rx_cmd, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(f"Remote receiver SSH PID: {receiver.pid}")
        time.sleep(args.tx_delay)
        print("Starting local transmitter...")
        tx_result = subprocess.run(tx_cmd, cwd=REPO_ROOT, env=local_env(), text=True, capture_output=True)
        if tx_result.stdout:
            sys.stdout.write(tx_result.stdout)
        write_stderr(tx_result.stderr)

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

    experiment_summary_path = run_validation_and_experiment(args, capture_dir, local_processed, tx_frame_copy, metadata_copy)

    if not args.keep_remote_capture:
        cleanup_cmd = [*ssh_base(args, remote_host), remote_cleanup_command(args, remote_capture)]
        run_checked(cleanup_cmd, cwd=REPO_ROOT)

    run_summary = {
        "capture_dir": str(capture_dir),
        "remote_host": remote_host,
        "remote_role": args.remote_role,
        "remote_os": args.remote_os,
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
