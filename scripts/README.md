# Script Index

Run scripts from the repository root with:

```bash
export PYTHONPATH="$PWD/src:$PWD/scripts"
```

## Result Generation

- `generate_icc2025_style_results.py`: Generates CSV tables and ICC-style plots from validated captures, optionally with offline AWGN.
- `plot_icc2025_style_results.py`: Regenerates plots from existing aggregate CSV files.
- `run_offline_awgn_experiment.py`: Adds AWGN to one validated capture and compares detectors.

## Detector Backends

- `PaperICL.py`: ICL detector adapter.
- `PaperDEFINED.py`: DEFINED decision-feedback detector adapter.
- `paper_transformer_backend.py`: Shared checkpoint loading and prompt construction.

## Validation and Diagnostics

- `qpsk_validate_from_post_costas.py`: Generic post-recovery validator for BPSK/QPSK/16QAM/64QAM captures.
- `receiver_experiment.py`: Runs detector backends on a validated capture.
- `diagnose_qpsk_transformer_permutations.py`: Single-capture QPSK class-order diagnostic.
- `batch_qpsk_transformer_permutation_diagnostic.py`: Multi-capture class-order diagnostic.

## Frame and SDR Collection

- `generate_defined_paper_frame.py`: Creates paper-style `T=31` frames.
- `run_defined_pilot_sweep.py`: Runs live SDR pilot-count sweeps.
- `run_mqam_headless_test.py`: Runs one live SDR transmit/receive/validate test.
- `run_distributed_mqam_test.py`: Runs one two-computer TX/RX test; defaults to local RX with remote Windows TX over SSH.
- `run_distributed_pilot_sweep.py`: Runs two-computer pilot sweeps; defaults to local RX with remote Windows TX over SSH and aggregates BER/SER/SNR/lock-score results.
- `Send_Signal_MQAM_Headless.py`: Headless GNU Radio transmitter.
- `Receive_Signal_MQAM_Headless.py`: Headless GNU Radio receiver.
