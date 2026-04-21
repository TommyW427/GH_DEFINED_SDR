# DEFINED-SDR Project Status

## Scope
This repo now has a working SDR validation path for repeated-pilot QPSK and a paper-style `T=31` symbol experiment path for pilot sweeps. The active recommendation is to use classical baselines first and defer Transformer/DEFINED model integration until trained checkpoints and a stable torch environment are available.

## Main Findings
1. The original QPSK BER post-processing path had a real payload-cycle ambiguity bug.
2. That bug was fixed, and BER improved substantially on previously captured data.
3. The live GNU Radio symbol-sync output remained poor even when the constellation visually locked.
4. Offline post-Costas fixed-phase decimation worked much better than the live symbol-sync stream.
5. The post-Costas validator is now the recommended ground-truth BER path.
6. For paper-style `T=31` QPSK experiments, SDR synchronization is strong and the hard case is low-pilot detection, not frame lock.

## Current Recommended Receiver Path
1. Capture oversampled processed samples.
2. Use the post-recovery validator with offline phase search.
3. Build aligned payload cases from the best phase.
4. Run detector baselines on those aligned payloads.

This is implemented by:
- [qpsk_validate_from_post_costas.py](/Users/tpdubya/Downloads/WIRELESS/DEFINED-SDR%20copy/qpsk_validate_from_post_costas.py)
- [receiver_experiment.py](/Users/tpdubya/Downloads/WIRELESS/DEFINED-SDR%20copy/receiver_experiment.py)

## Key Scripts

### QPSK Headless
- [Send_Signal_QPSK_Headless.py](/Users/tpdubya/Downloads/WIRELESS/DEFINED-SDR%20copy/Send_Signal_QPSK_Headless.py)
- [Receive_Signal_QPSK_Headless.py](/Users/tpdubya/Downloads/WIRELESS/DEFINED-SDR%20copy/Receive_Signal_QPSK_Headless.py)
- [run_qpsk_headless_test.py](/Users/tpdubya/Downloads/WIRELESS/DEFINED-SDR%20copy/run_qpsk_headless_test.py)

### Generic M-QAM
- [core/modulation.py](/Users/tpdubya/Downloads/WIRELESS/DEFINED-SDR%20copy/core/modulation.py)
- [generate_mqam_frame.py](/Users/tpdubya/Downloads/WIRELESS/DEFINED-SDR%20copy/generate_mqam_frame.py)
- [Send_Signal_MQAM_Headless.py](/Users/tpdubya/Downloads/WIRELESS/DEFINED-SDR%20copy/Send_Signal_MQAM_Headless.py)
- [Receive_Signal_MQAM_Headless.py](/Users/tpdubya/Downloads/WIRELESS/DEFINED-SDR%20copy/Receive_Signal_MQAM_Headless.py)
- [run_mqam_headless_test.py](/Users/tpdubya/Downloads/WIRELESS/DEFINED-SDR%20copy/run_mqam_headless_test.py)

### Paper-Style `T=31` Experiments
- [generate_defined_paper_frame.py](/Users/tpdubya/Downloads/WIRELESS/DEFINED-SDR%20copy/generate_defined_paper_frame.py)
- [run_defined_pilot_sweep.py](/Users/tpdubya/Downloads/WIRELESS/DEFINED-SDR%20copy/run_defined_pilot_sweep.py)

### Detector Abstraction
- [core/frame_processing/detector_interfaces.py](/Users/tpdubya/Downloads/WIRELESS/DEFINED-SDR%20copy/core/frame_processing/detector_interfaces.py)
- [core/frame_processing/receiver_experiment.py](/Users/tpdubya/Downloads/WIRELESS/DEFINED-SDR%20copy/core/frame_processing/receiver_experiment.py)

## Current Baselines
Active, recommended baselines:
- `mmse`
- `mmse_df`

These are available through [receiver_experiment.py](/Users/tpdubya/Downloads/WIRELESS/DEFINED-SDR%20copy/receiver_experiment.py) and [run_defined_pilot_sweep.py](/Users/tpdubya/Downloads/WIRELESS/DEFINED-SDR%20copy/run_defined_pilot_sweep.py).

## Transformer Status
The repo has paper-backend wrappers:
- [PaperICL.py](/Users/tpdubya/Downloads/WIRELESS/DEFINED-SDR%20copy/PaperICL.py)
- [PaperDEFINED.py](/Users/tpdubya/Downloads/WIRELESS/DEFINED-SDR%20copy/PaperDEFINED.py)
- [paper_transformer_backend.py](/Users/tpdubya/Downloads/WIRELESS/DEFINED-SDR%20copy/paper_transformer_backend.py)

But they are currently not part of the recommended test flow because:
1. there are no trained checkpoints in the repo
2. the current Miniconda torch installation reports a NumPy ABI mismatch warning on import

So for now, sweeps should skip `icl` and `defined`.

## Paper Alignment
From `Project 2 Paper ICC 2025.pdf`:
- coherence/frame length is at most `T = 31` symbols
- first `k` symbols are pilots
- payload length is `31 - k` symbols

The paper-style SDR frame after sync preamble is now:
- `[k pilot symbols] + [31-k data symbols]`

## Fast Test Defaults
The default capture timings were shortened to make repetitions and sweeps practical.

### `run_mqam_headless_test.py`
- `rx-time = 6.0`
- `tx-delay = 1.5`
- default `tx-time = max(2.5, rx_time - tx_delay - 1.0)`

### `run_defined_pilot_sweep.py`
- `rx-time = 6.0`
- `tx-delay = 1.5`
- default detectors: `mmse mmse_df`

### `run_qpsk_headless_test.py`
- `rx-time = 8.0`
- `tx-delay = 2.0`
- default `tx-time = max(3.0, rx_time - tx_delay - 1.0)`

These are intended to preserve enough margin for SDR startup while dramatically reducing sweep runtime.

## Recommended Commands

### Fast paper-style QPSK sweep
```bash
/Users/tpdubya/miniconda/bin/python3 run_defined_pilot_sweep.py \
  --python /Users/tpdubya/miniconda/bin/python3 \
  --modulation QPSK \
  --pilot-symbols 1 2 3 4 6 8 \
  --repetitions 5 \
  --rx-device-args 'driver=plutosdr,uri=usb:20.3.5' \
  --tx-device-args 'driver=plutosdr,uri=usb:20.4.5' \
  --carrier-recovery costas
```

### Fast single generic M-QAM test
```bash
/Users/tpdubya/miniconda/bin/python3 run_mqam_headless_test.py \
  --python /Users/tpdubya/miniconda/bin/python3 \
  --modulation 16QAM \
  --rx-device-args 'driver=plutosdr,uri=usb:20.3.5' \
  --tx-device-args 'driver=plutosdr,uri=usb:20.4.5' \
  --run-experiment
```

### QPSK post-recovery validation on an existing capture
```bash
python3 qpsk_validate_from_post_costas.py \
  --capture-dir 'captures/qpsk_headless_20260417_144657'
```

## Recent Paper-Style Sweep Result
From the completed QPSK sweep at `T=31`:
- `k=1`: BER around `0.38`
- `k=2`: BER around `0.50`
- `k>=4`: BER `0.0` on that sweep

Interpretation:
- frame lock is strong across pilot counts
- the low-pilot regime is the real challenge
- repetitions and SER aggregation are now the right next step for a more defensible paper-style result

## Output Files From Sweeps
Each sweep now writes:
- `defined_pilot_sweep_summary.json`
- `defined_pilot_sweep_summary.csv`
- `defined_pilot_sweep_detailed.csv`

The detector experiment outputs now include both BER and SER.
The post-recovery validator and sweep summaries now also include an empirical
SNR estimate derived from pilot-fit residuals.

## Next Recommended Work
1. Run repeated QPSK paper sweeps with the faster defaults.
2. Compare `mmse` and `mmse_df` using mean BER and mean SER with standard deviations.
3. Only revisit `PaperICL` and `PaperDEFINED` after real checkpoints and a working torch environment are available.
