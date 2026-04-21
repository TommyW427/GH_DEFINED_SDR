# Distributed Two-Computer PlutoSDR Pipeline

This pipeline lets one controller computer run a full transmit/receive sweep while a second computer only hosts the receiver PlutoSDR. The operator launches one command on the TX/controller computer; the script starts the RX flowgraph remotely over SSH, runs TX locally, copies the remote capture back, validates it, and aggregates BER/SER/SNR/lock-score results.

## Assumptions

- Both computers have this repository checked out.
- Both computers have GNU Radio, SoapySDR, and PlutoSDR drivers installed.
- The TX/controller computer can SSH into the RX computer without an interactive password prompt.
- The TX PlutoSDR is attached to the controller computer.
- The RX PlutoSDR is attached to the remote receiver computer.
- Run commands from the repository root with `PYTHONPATH="$PWD/src:$PWD/scripts"`.

If the repository path differs between computers, pass `--remote-repo` explicitly.

## One-Time SSH Setup

From the controller/TX computer:

```bash
ssh-keygen -t ed25519
ssh-copy-id user@rx-computer.local
ssh user@rx-computer.local "cd /path/to/GH_DEFINED_SDR && pwd"
```

On macOS, `ssh-copy-id` may need to be installed separately. You can also append the public key in `~/.ssh/id_ed25519.pub` to the receiver computer's `~/.ssh/authorized_keys`.

## Single Distributed Run

First create a frame locally:

```bash
export PYTHONPATH="$PWD/src:$PWD/scripts"

python scripts/generate_defined_paper_frame.py \
  --modulation QPSK \
  --pilot-symbols 4 \
  --total-symbols 31 \
  --frame-output frames/distributed_qpsk_k04.bin \
  --payload-output payloads_qpsk/distributed_payload_k04.bin
```

Then run one distributed capture:

```bash
python scripts/run_distributed_mqam_test.py \
  --modulation QPSK \
  --tx-frame frames/distributed_qpsk_k04.bin \
  --metadata frames/distributed_qpsk_k04.json \
  --remote-rx-host user@rx-computer.local \
  --remote-repo "/path/to/GH_DEFINED_SDR" \
  --python /Users/tpdubya/miniconda/bin/python3 \
  --remote-python /Users/tpdubya/miniconda/bin/python3 \
  --rx-device-args 'driver=plutosdr' \
  --tx-device-args 'driver=plutosdr' \
  --rx-time 6 \
  --tx-delay 1.5 \
  --tx-time 3.5 \
  --carrier-recovery none \
  --run-experiment
```

Use `--dry-run` to print the SSH, TX, and SCP commands without touching either radio.

## Distributed Pilot Sweep

Launch this on the TX/controller computer:

```bash
export PYTHONPATH="$PWD/src:$PWD/scripts"

python scripts/run_distributed_pilot_sweep.py \
  --modulation QPSK \
  --pilot-symbols 1 2 4 8 16 24 30 \
  --repetitions 3 \
  --remote-rx-host user@rx-computer.local \
  --remote-repo "/path/to/GH_DEFINED_SDR" \
  --python /Users/tpdubya/miniconda/bin/python3 \
  --remote-python /Users/tpdubya/miniconda/bin/python3 \
  --rx-device-args 'driver=plutosdr' \
  --tx-device-args 'driver=plutosdr' \
  --rx-time 6 \
  --tx-delay 1.5 \
  --tx-time 3.5 \
  --inter-run-delay 1 \
  --lock-threshold 0.95 \
  --exclude-bad-lock \
  --detectors mmse mmse_df
```

Outputs are written under:

```text
experiments/<sweep-tag>/
captures/<run-tag>/
```

The sweep writes:

- `distributed_pilot_sweep_summary.json`
- `distributed_pilot_sweep_summary.csv`
- `distributed_pilot_sweep_detailed.csv`

Each run capture contains:

- `transmitted_frame.bin`
- `frame_metadata.json`
- `received_processed_<modulation>.bin`
- `received_iq_raw_<modulation>.bin`
- `mqam_validation.json`
- `mqam_validation.npz`
- `receiver_experiment.json`
- `distributed_run_metadata.json`

## Device Argument Notes

If each computer has only one PlutoSDR attached, `driver=plutosdr` is usually enough. If either computer sees multiple radios, use the full URI on that machine:

```bash
--rx-device-args 'driver=plutosdr,uri=usb:20.3.5'
--tx-device-args 'driver=plutosdr,uri=usb:20.4.5'
```

The RX URI is interpreted on the remote RX computer. The TX URI is interpreted on the local controller computer.

## Practical Timing

The default timing is intentionally short:

- `--rx-time 6`
- `--tx-delay 1.5`
- `--tx-time 3.5`

Increase `--rx-time` or `--tx-delay` if the remote receiver needs longer to initialize. If lock scores become inconsistent, first try increasing `--tx-delay` to `2.0` or `3.0` before increasing the entire run length.
