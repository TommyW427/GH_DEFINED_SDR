# ICC 2025 Style SDR Results

- Modulation: `QPSK`
- Target offline AWGN SNR: `5 dB`
- Repetitions per capture: `20`
- Selected captures: `24`

## Tables

- `clean`: `clean.csv`
- `noisy_summary`: `noisy_summary.csv`
- `noisy_steps`: `noisy_steps.csv`
- `snr`: `snr.csv`
- `aggregate_by_pilot`: `aggregate_by_pilot.csv`
- `aggregate_by_context`: `aggregate_by_context.csv`

## Figures

- `ser_vs_pilots.png`
- `ser_vs_pilots.pdf`
- `ber_vs_pilots.png`
- `ber_vs_pilots.pdf`
- `ser_vs_context_k02.png`
- `ser_vs_context_k02.pdf`

## Method Note

These plots intentionally isolate detector behavior by adding AWGN after SDR synchronization/alignment. They should be reported separately from fully empirical over-the-air SNR sweeps.
