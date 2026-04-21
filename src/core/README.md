# Core Modules

This directory contains the core processing modules for DEFINED-SDR, organized by functionality.

## Structure

```
core/
├── synchronization/     # Signal synchronization modules
│   ├── agc.py          # Automatic Gain Control
│   ├── costas_loop.py  # Costas Loop for carrier sync
│   ├── timing_recovery.py  # Mueller & Muller timing recovery
│   └── pipeline.py     # Complete synchronization pipeline
│
├── frame_processing/    # Frame-level operations
│   ├── config.py       # FrameConfig dataclass
│   ├── synchronizer.py # FrameSynchronizer class
│   ├── channel_estimator.py  # ChannelEstimator class
│   └── symbol_detector.py    # SymbolDetector class
│
└── validation/          # BER/SER validation
    └── ber_validator.py  # validate_frame function
```

## Module Descriptions

### synchronization/
Signal synchronization components for processing raw IQ data.

- **agc.py**: Normalizes signal amplitude using automatic gain control
- **costas_loop.py**: Corrects carrier frequency and phase offsets
- **timing_recovery.py**: Symbol timing synchronization using Mueller & Muller TED
- **pipeline.py**: Complete AGC → Costas → Timing pipeline with `synchronize_iq()` convenience function

### frame_processing/
Frame-level processing for channel estimation and symbol detection.

- **config.py**: `FrameConfig` dataclass defining frame structure
- **synchronizer.py**: `FrameSynchronizer` for finding frame boundaries and extracting regions
- **channel_estimator.py**: `ChannelEstimator` supporting LS, MMSE, and Transformer methods
- **symbol_detector.py**: `SymbolDetector` supporting MRC, ML, and Transformer methods

### validation/
BER/SER validation tools.

- **ber_validator.py**: Simple validation function for received frames

## Usage Examples

### Using Synchronization
```python
from core.synchronization import synchronize_iq
import numpy as np

raw_iq = np.fromfile('received_iq.bin', dtype=np.complex64)
symbols, info = synchronize_iq(raw_iq, samples_per_symbol=25, modulation='BPSK')
```

### Using Frame Processing
```python
from core.frame_processing import FrameConfig, FrameSynchronizer

config = FrameConfig(preamble_bits=2000, pilot_bits=64, data_bits=64)
sync = FrameSynchronizer(config)
pilot_pos, phase_inv, errors = sync.find_pilot(rx_symbols, pilot_bits)
```

### Using Channel Estimation
```python
from core.frame_processing import ChannelEstimator, FrameConfig

config = FrameConfig()
estimator = ChannelEstimator(config)
H_hat = estimator.estimate_MMSE(pilot_rx, pilot_tx)
```

## Backwards Compatibility

The original top-level files (`synchronization.py`, `channel_analysis.py`, `validate_ber.py`)
still work and now import from this modular structure. This ensures existing code continues
to function without modifications.
