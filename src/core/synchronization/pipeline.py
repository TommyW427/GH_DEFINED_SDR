#!/usr/bin/env python3
"""
Complete Synchronization Pipeline
==================================

Combines AGC, Costas Loop, and Timing Recovery into a complete pipeline.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from .agc import AGC
from .costas_loop import CostasLoop
from .timing_recovery import MuellerMullerTimingRecovery


@dataclass
class SynchronizationConfig:
    """Configuration for synchronization blocks"""
    # AGC parameters
    agc_rate: float = 1e-4          # AGC adaptation rate (smaller = slower)
    agc_reference: float = 1.0      # Target amplitude
    agc_max_gain: float = 65536     # Maximum AGC gain

    # Costas loop parameters
    costas_loop_bw: float = 0.0628  # Loop bandwidth (normalized)
    costas_damping: float = 0.707   # Damping factor (sqrt(2)/2 for critical damping)
    costas_order: int = 2           # 2 for BPSK, 4 for QPSK

    # Symbol timing recovery parameters
    timing_loop_bw: float = 0.01    # Timing loop bandwidth (normalized) - lower for stability
    timing_damping: float = 1.0     # Damping factor
    timing_max_deviation: float = 0.5  # Max deviation in symbols - tighter bounds

    # Frame parameters
    samples_per_symbol: int = 25    # Oversampling rate


class Synchronizer:
    """
    Complete synchronization pipeline for raw IQ data

    Pipeline:
        Raw IQ → AGC → Costas Loop → Symbol Timing Recovery → Synchronized Symbols

    This replicates GNU Radio's synchronization chain in pure Python.
    """

    def __init__(self, config: SynchronizationConfig):
        """
        Initialize synchronizer

        Args:
            config: Synchronization configuration
        """
        self.config = config

        # Create synchronization blocks
        self.agc = AGC(
            rate=config.agc_rate,
            reference=config.agc_reference,
            max_gain=config.agc_max_gain
        )

        self.costas = CostasLoop(
            loop_bw=config.costas_loop_bw,
            damping=config.costas_damping,
            order=config.costas_order
        )

        self.timing_recovery = MuellerMullerTimingRecovery(
            sps=config.samples_per_symbol,
            loop_bw=config.timing_loop_bw,
            damping=config.timing_damping,
            max_deviation=config.timing_max_deviation
        )

    def synchronize(
        self,
        raw_iq: np.ndarray,
        return_intermediate: bool = False
    ) -> Tuple[np.ndarray, Optional[dict]]:
        """
        Synchronize raw IQ samples

        Args:
            raw_iq: Raw IQ samples (complex64)
            return_intermediate: If True, return intermediate results for debugging

        Returns:
            Tuple of:
                - Synchronized symbols (1 sample per symbol)
                - Optional dict with intermediate results (if return_intermediate=True)
        """
        print(f"[Sync] Input: {len(raw_iq):,} samples")

        # Step 1: AGC
        agc_output = self.agc.process(raw_iq)
        print(f"[Sync] AGC complete")

        # Step 2: Costas Loop (carrier synchronization)
        costas_output = self.costas.process(agc_output)
        print(f"[Sync] Costas loop complete (phase: {self.costas.phase:.4f} rad, "
              f"freq: {self.costas.freq:.6f})")

        # Step 3: Symbol Timing Recovery
        symbols = self.timing_recovery.process(costas_output)
        print(f"[Sync] Timing recovery complete: {len(symbols):,} symbols "
              f"(omega: {self.timing_recovery.omega:.4f} sps, expected: {self.config.samples_per_symbol})")

        # Warn if omega is far from expected
        omega_error = abs(self.timing_recovery.omega - self.config.samples_per_symbol)
        if omega_error > 2.0:
            print(f"[Sync] WARNING: Timing omega deviated significantly from expected!")
            print(f"       This may indicate timing recovery failed to converge.")
            print(f"       Try: use_synchronization=False for simple decimation")

        # Prepare intermediate results if requested
        intermediate = None
        if return_intermediate:
            intermediate = {
                'agc_output': agc_output,
                'costas_output': costas_output,
                'agc_gain': self.agc.gain,
                'costas_phase': self.costas.phase,
                'costas_freq': self.costas.freq,
                'timing_omega': self.timing_recovery.omega
            }

        return symbols, intermediate

    def reset(self):
        """Reset all synchronization blocks to initial state"""
        self.agc.gain = 1.0
        self.costas.phase = 0.0
        self.costas.freq = 0.0
        self.timing_recovery.mu = 0.0
        self.timing_recovery.omega = self.config.samples_per_symbol
        self.timing_recovery.prev_sample = 0j
        self.timing_recovery.prev_symbol = 0j
        self.timing_recovery.prev_prev_symbol = 0j


def synchronize_iq(
    raw_iq: np.ndarray,
    samples_per_symbol: int = 25,
    modulation: str = "BPSK",
    return_intermediate: bool = False
) -> Tuple[np.ndarray, Optional[dict]]:
    """
    Convenience function to synchronize raw IQ data

    Args:
        raw_iq: Raw IQ samples (complex64)
        samples_per_symbol: Oversampling rate
        modulation: "BPSK" or "QPSK"
        return_intermediate: Return intermediate processing results

    Returns:
        Tuple of:
            - Synchronized symbols
            - Optional intermediate results dict

    Example:
        >>> raw_iq = np.fromfile('received_iq_raw.bin', dtype=np.complex64)
        >>> symbols, _ = synchronize_iq(raw_iq, samples_per_symbol=25)
        >>> print(f"Recovered {len(symbols)} symbols")
    """
    # Create configuration
    config = SynchronizationConfig(
        samples_per_symbol=samples_per_symbol,
        costas_order=2 if modulation == "BPSK" else 4
    )

    # Create synchronizer and process
    synchronizer = Synchronizer(config)
    return synchronizer.synchronize(raw_iq, return_intermediate=return_intermediate)
