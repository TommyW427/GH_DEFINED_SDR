# Distributed Two-Computer PlutoSDR Pipeline

This pipeline lets one controller computer run a full transmit/receive sweep while a second computer hosts the transmitter PlutoSDR. For your setup, the SSH target is the Windows TX computer:

```text
C:\Users\tpdub\Downloads\Wireless\GH_DEFINED_SDR
```

The operator launches one command on the RX/controller computer. The script starts the local RX flowgraph, copies the frame to the Windows TX computer, starts TX over SSH, validates the local RX capture, and aggregates BER/SER/SNR/lock-score results.

## Assumptions

- Both computers have this repository checked out.
- The RX/controller computer has the RX PlutoSDR attached.
- The Windows SSH computer has the TX PlutoSDR attached.
- Both computers have GNU Radio, SoapySDR, and PlutoSDR drivers installed.
- The Windows TX computer has a conda environment named `gnuradio`.
- The RX/controller computer can SSH into the Windows TX computer without an interactive password prompt.
- Run commands on the RX/controller computer from the repository root with `PYTHONPATH="$PWD/src:$PWD/scripts"`.

The scripts use forward-slash Windows paths internally for SSH/SCP:

```text
C:/Users/tpdub/Downloads/Wireless/GH_DEFINED_SDR
```

## One-Time SSH Setup

From the RX/controller computer:

```bash
ssh-keygen -t ed25519
ssh tpdub@windows-hostname "powershell.exe -NoProfile -Command \"Write-Host ok\""
ssh tpdub@windows-hostname "powershell.exe -NoProfile -Command \"Set-Location 'C:/Users/tpdub/Downloads/Wireless/GH_DEFINED_SDR'; Get-Location\""
ssh tpdub@windows-hostname "powershell.exe -NoProfile -Command \"(& conda shell.powershell hook) | Out-String | Invoke-Expression; conda activate gnuradio; python -c 'import gnuradio; print(123)'\""
```

If key-based login is not configured, add your controller computer's public key to the Windows user's `authorized_keys` file.

Or run the setup helper from the Mac/controller. It will prompt for the Windows password once, install your public key, and verify passwordless SSH:

```bash
python scripts/setup_windows_ssh_key.py \
  --host tpdub@10.41.1.254
```

If you use a non-default SSH port, add `--ssh-port <port>`.

## Single Distributed Run

Keep the two repositories synchronized with git before running the distributed pipeline.

First create a frame locally on the RX/controller computer:

```bash
export PYTHONPATH="$PWD/src:$PWD/scripts"

python scripts/generate_defined_paper_frame.py \
  --modulation QPSK \
  --pilot-symbols 4 \
  --total-symbols 31 \
  --frame-output frames/distributed_qpsk_k04.bin \
  --payload-output payloads_qpsk/distributed_payload_k04.bin
```

Then run one distributed capture. Replace `tpdub@windows-hostname` with the actual SSH target:

```bash
python scripts/run_distributed_mqam_test.py \
  --modulation QPSK \
  --tx-frame frames/distributed_qpsk_k04.bin \
  --metadata frames/distributed_qpsk_k04.json \
  --remote-role tx \
  --remote-os windows \
  --remote-tx-host tpdub@windows-hostname \
  --remote-repo "C:/Users/tpdub/Downloads/Wireless/GH_DEFINED_SDR" \
  --python /Users/tpdubya/miniconda/bin/python3 \
  --remote-python python \
  --remote-conda-env gnuradio \
  --rx-device-args 'driver=plutosdr' \
  --tx-device-args 'driver=plutosdr' \
  --rx-time 6 \
  --tx-delay 1.5 \
  --tx-time 3.5 \
  --carrier-recovery none \
  --run-experiment \
  --experiment-detectors mmse mmse_df icl defined
```

Use `--dry-run` to print the local RX, SCP, and remote TX commands without touching either radio.

## Distributed Pilot Sweep

Launch this on the RX/controller computer:

```bash
export PYTHONPATH="$PWD/src:$PWD/scripts"

python scripts/run_distributed_pilot_sweep.py \
  --modulation QPSK \
  --pilot-symbols 1 2 4 8 16 24 30 \
  --repetitions 3 \
  --remote-role tx \
  --remote-os windows \
  --remote-tx-host tpdub@windows-hostname \
  --remote-repo "C:/Users/tpdub/Downloads/Wireless/GH_DEFINED_SDR" \
  --python /Users/tpdubya/miniconda/bin/python3 \
  --remote-python python \
  --remote-conda-env gnuradio \
  --rx-device-args 'driver=plutosdr' \
  --tx-device-args 'driver=plutosdr' \
  --rx-time 6 \
  --tx-delay 1.5 \
  --tx-time 3.5 \
  --inter-run-delay 1 \
  --lock-threshold 0.95 \
  --retry-until-lock \
  --max-lock-attempts 5 \
  --retry-delay 1 \
  --exclude-bad-lock \
  --detectors mmse mmse_df icl defined
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

## Lock Retry Policy

By default, the distributed sweep retries each logical repetition until the selected validator lock score reaches `--lock-threshold` or the max attempt count is reached:

```bash
--retry-until-lock \
--max-lock-attempts 5 \
--retry-delay 1
```

Use `--max-lock-attempts 0` for unlimited retries. That is useful for unattended collection, but it can run forever if the radios or synchronization path are unhealthy.

The summary CSV includes:

- `total_attempts`: total captures attempted for that pilot count.
- `attempts_per_selected_run_mean`: average number of attempts needed for each accepted repetition.
- `attempts_per_selected_run_max`: worst-case attempts needed for an accepted repetition.
- `sync_source`, `phase`, `frame_start`, `pilot_start`, `payload_cycle_offset`, and `frame_lock_pilots_checked`: synchronization diagnostics for the accepted runs.

The detailed CSV includes one row per selected repetition. The JSON includes both selected `runs` and all attempted `attempts`, so bad-lock attempts remain auditable.

## Distance And Lock Quality

Moving the radios farther apart can help if close-range operation is saturating the RX front end, but after that point it usually hurts synchronization. Lock score is a correlation/alignment metric, so it is sensitive to SNR, CFO/phase drift, timing jitter, multipath, and AGC settling. You can see which failure mode dominates by comparing these fields in the detailed CSV:

- `validation_mean_snr_db`: if this drops with distance, the receiver is simply becoming too weak/noisy for reliable preamble/pilot correlation.
- `validation_mean_pilot_ber`: if this is high when lock is low, the pilots are not being aligned/equalized reliably.
- `sync_source`: if it flips between `sync_tail`, `preamble`, and `pilot`, the synchronizer is not consistently finding the same frame boundary.
- `phase`: if accepted runs use many different sample phases, timing/phase selection is unstable.
- `frame_start` and `pilot_start`: large jumps between attempts indicate false correlation peaks rather than a stable capture.

For distance sweeps, keep TX/RX gain fixed and record physical distance. The useful range is typically the middle region: far enough to avoid overload, but close enough that the preamble/tail still gives a strong correlation peak.

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

The RX URI is interpreted on the local RX/controller computer. The TX URI is interpreted on the remote Windows TX computer.

## Remote SDR Preflight

Before launching the remote radio process, the pipeline now runs a remote SoapySDR preflight inside the activated `gnuradio` environment. It prints:

- SoapySDR API version.
- SoapySDR root and plugin search paths.
- Loaded SoapySDR modules.
- Enumerated SDR devices.
- Whether the requested `--tx-device-args` or `--rx-device-args` can be opened.

If you see `SoapySDR::Device::make() no match`, use the device string printed in `SoapySDR devices`. For example, if the preflight reports:

```json
{"driver": "plutosdr", "uri": "usb:20.4.5"}
```

then run with:

```bash
--tx-device-args 'driver=plutosdr,uri=usb:20.4.5'
```

If the preflight lists no Pluto device, the issue is on the Windows TX side: Radioconda is active, but SoapySDR cannot see the Pluto plugin or USB device. Check that Windows can run:

```powershell
conda activate gnuradio
python -c "import SoapySDR, json; print(json.dumps([dict(d) for d in SoapySDR.Device.enumerate()], indent=2))"
```

You can disable this check with `--no-remote-sdr-preflight`, but that is only useful after the device args are known good.

## Remote Conda Notes

The distributed scripts activate the remote conda environment before launching GNU Radio:

```bash
--remote-conda-env gnuradio
```

If `conda` is not on PATH for non-interactive SSH sessions, the launcher checks common Windows install paths such as:

```text
C:/Users/<user>/radioconda/condabin/conda.bat
C:/Users/<user>/radioconda/Scripts/conda.exe
C:/Users/<user>/miniconda3/Scripts/conda.exe
C:/Users/<user>/anaconda3/Scripts/conda.exe
C:/Users/<user>/miniforge3/Scripts/conda.exe
C:/ProgramData/miniconda3/Scripts/conda.exe
```

If your install is elsewhere, pass the full path to the conda executable:

```bash
--remote-conda-exe "C:/Users/tpdub/radioconda/condabin/conda.bat"
```

Keep `--remote-python python` when using conda activation; after activation, `python` resolves to the environment interpreter.

If your actual Windows path is `C:\Uers\tpdub\...` rather than the usual `C:\Users\tpdub\...`, pass that exact spelling with forward slashes:

```bash
--remote-conda-exe "C:/Uers/tpdub/radioconda/condabin/conda.bat"
```

The remote launcher also verifies the environment before transmitting:

```powershell
python -c "import gnuradio"
```

## Practical Timing

The default timing is intentionally short:

- `--rx-time 6`
- `--tx-delay 1.5`
- `--tx-time 3.5`

Increase `--tx-delay` to `2.0` or `3.0` if the local receiver needs more time to settle before the remote TX starts. Increase `--rx-time` only if the receiver is stopping before the remote TX finishes.

## Inverse Topology

The scripts still support the inverse topology, where the controller hosts TX and the remote computer hosts RX:

```bash
python scripts/run_distributed_mqam_test.py \
  --remote-role rx \
  --remote-os posix \
  --remote-rx-host user@rx-host \
  ...
```

For your current Windows setup, use `--remote-role tx`.
