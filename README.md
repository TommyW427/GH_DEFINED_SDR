# DEFINED SDR Experiments

This repository contains a clean, reproducible subset of the SDR experiment code used to replicate and extend the ICC 2025 DEFINED receiver experiments with PlutoSDR captures.

The focus is on paper-style block-fading symbol detection:

- Frame format: `T = 31` symbols after an SDR synchronization preamble.
- Pilot sweep: first `k` symbols are known pilots, remaining `31-k` are payload symbols.
- Detectors: `MMSE`, `MMSE-DF`, `DEFINED-ICL`, and `DEFINED-DF`.
- Reported metrics: BER, SER, lock score, SNR, and context-length curves.
- Included generated example: QPSK at 5 dB offline AWGN using validated SDR captures.

## Repository Layout

```text
.
├── docs/                 # ICC paper PDF and project status notes
├── models/               # Trained BPSK/QPSK ICL and DEFINED checkpoints
├── results/              # Generated CSV tables and paper-style plots
├── scripts/              # Experiment, validation, plotting, and SDR scripts
└── src/core/             # Shared modulation and detector interfaces
```

Raw SDR captures, generated binary frames, Python caches, and ad-hoc debugging files are intentionally excluded.

## Quick Start

Create an environment with the Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Most scripts expect both `src/` and `scripts/` on `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/src:$PWD/scripts"
```

Regenerate plots from the included result CSVs:

```bash
python3 scripts/plot_icc2025_style_results.py \
  --results-dir results/qpsk_snr5_reps20 \
  --target-snr-db 5 \
  --context-pilots 1 2 4 8
```

Generate a new ICC-style offline AWGN result set from validated captures:

```bash
python3 scripts/generate_icc2025_style_results.py \
  --modulation QPSK \
  --pilot-symbols 1 2 3 4 6 8 \
  --context-pilot 2 \
  --target-snr-db 5 \
  --repetitions 20 \
  --max-captures-per-k 4 \
  --lock-threshold 0.95 \
  --max-clean-validation-ber 0 \
  --detectors mmse mmse_df icl defined \
  --output-dir results/qpsk_snr5_new
```

That command requires validated capture folders. The included `results/` folder is already generated, so new captures are not required to view plots.

## Included Result Set

The included result set is:

```text
results/qpsk_snr5_reps20
```

It contains:

- `aggregate_by_pilot.csv`: BER/SER vs number of pilot symbols.
- `aggregate_by_context.csv`: BER/SER vs context length.
- `clean.csv`: clean validated detector performance before AWGN injection.
- `noisy_summary.csv`: repetition-level detector results after AWGN injection.
- `snr.csv`: actual synthetic SNR and pilot-estimated SNR metadata.
- `*.png` and `*.pdf`: publication-style plots.

Important method note: the included plots add AWGN after SDR synchronization/alignment. This isolates detector and channel-estimation performance and should be reported separately from fully empirical over-the-air SNR sweeps where synchronization can also fail.

## Model Checkpoints

The checkpoints in `models/` are loaded automatically by `scripts/paper_transformer_backend.py` when running from repository root with `PYTHONPATH="$PWD/src:$PWD/scripts"`.

Default checkpoint filenames:

```text
models/trained_model_ICL_QPSK.pth
models/trained_model_DEFINED_QPSK.pth
models/trained_model_ICL_BPSK.pth
models/trained_model_DEFINED_BPSK.pth
```

You can override checkpoint paths with:

```bash
export PAPER_ICL_CHECKPOINT=/path/to/checkpoint.pth
export PAPER_DEFINED_CHECKPOINT=/path/to/checkpoint.pth
```

The QPSK checkpoint class mapping is configured in `paper_transformer_backend.py`; it defaults to `0,2,1,3` because diagnostics showed the checkpoints use natural Cartesian QPSK ordering while the SDR modem uses Gray bit ordering. Override with:

```bash
export PAPER_CLASS_PERM_QPSK=0,2,1,3
```

## Hardware Scripts

The SDR scripts are included for completeness:

- `scripts/Send_Signal_MQAM_Headless.py`
- `scripts/Receive_Signal_MQAM_Headless.py`
- `scripts/run_mqam_headless_test.py`
- `scripts/run_defined_pilot_sweep.py`

They require GNU Radio, SoapySDR, and PlutoSDR drivers. The plotting and offline AWGN pipelines do not require SDR hardware.

## Known Environment Notes

The original experiments used a Conda Python with GNU Radio/SoapySDR. Transformer inference requires PyTorch. In the original environment, PyTorch emitted a NumPy ABI warning but still completed runs when imported before NumPy; the scripts that need Torch handle that import order.

For a cleaner environment, use compatible PyTorch and NumPy versions.

## What Is Excluded

The clean repository intentionally excludes:

- Raw SDR capture directories.
- Generated frame and payload binary files.
- Debug scripts and temporary notebooks/plots.
- `.grc` GNU Radio design files.
- `__pycache__`, `.DS_Store`, and profiling artifacts.

Those files are not needed to reproduce the included plots or understand the experiment pipeline.
