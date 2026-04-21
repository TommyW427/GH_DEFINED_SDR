#!/usr/bin/env python3
"""
Costas Loop for Carrier Synchronization
========================================

Corrects carrier frequency offset and phase noise for BPSK/QPSK signals.
"""

import numpy as np


class CostasLoop:
    """
    Costas Loop for Carrier Phase and Frequency Synchronization

    Corrects:
        - Carrier frequency offset (CFO)
        - Carrier phase offset
        - Phase noise/drift

    Algorithm (for BPSK, order=2):
        error = real(y) * imag(y)  [phase detector]
        phase += loop_filter(error)
        y_out = y * exp(-j * phase)

    For QPSK (order=4):
        error = sign(real(y)) * imag(y) - sign(imag(y)) * real(y)

    The loop bandwidth controls the tradeoff:
        - Higher BW: faster acquisition, more noise
        - Lower BW: slower acquisition, better tracking
    """

    def __init__(
        self,
        loop_bw: float = 0.0628,
        damping: float = 0.707,
        order: int = 2
    ):
        """
        Initialize Costas loop

        Args:
            loop_bw: Loop bandwidth (normalized to sample rate)
            damping: Damping factor (0.707 = critically damped)
            order: 2 for BPSK, 4 for QPSK
        """
        self.order = order
        self.phase = 0.0  # Current phase estimate
        self.freq = 0.0   # Current frequency offset estimate

        # Compute loop filter coefficients from loop bandwidth and damping
        # These determine how aggressively the loop tracks phase errors
        denom = 1.0 + 2.0 * damping * loop_bw + loop_bw * loop_bw
        self.alpha = (4 * damping * loop_bw) / denom  # Proportional gain
        self.beta = (4 * loop_bw * loop_bw) / denom   # Integral gain

    def _phase_detector(self, sample: np.complex128) -> float:
        """
        Phase error detector

        For BPSK (order=2): error = real(y) * imag(y)
        For QPSK (order=4): error = sign(real) * imag - sign(imag) * real

        Args:
            sample: Input sample (after rotation)

        Returns:
            Phase error estimate
        """
        if self.order == 2:
            # BPSK phase detector
            error = np.real(sample) * np.imag(sample)
        elif self.order == 4:
            # QPSK phase detector
            error = (np.sign(np.real(sample)) * np.imag(sample) -
                    np.sign(np.imag(sample)) * np.real(sample))
        else:
            raise ValueError(f"Unsupported Costas loop order: {self.order}")

        return error

    def process(self, samples: np.ndarray) -> np.ndarray:
        """
        Apply Costas loop to input samples

        Args:
            samples: Complex input samples (AGC-normalized)

        Returns:
            Carrier-synchronized samples
        """
        output = np.zeros_like(samples, dtype=np.complex64)

        for i, x in enumerate(samples):
            # Rotate sample by current phase estimate (remove carrier offset)
            y = x * np.exp(-1j * self.phase)
            output[i] = y

            # Compute phase error
            error = self._phase_detector(y)

            # Update frequency estimate (integral path)
            self.freq += self.beta * error

            # Update phase estimate (proportional + integral)
            self.phase += self.alpha * error + self.freq

            # Wrap phase to [-π, π]
            while self.phase > np.pi:
                self.phase -= 2 * np.pi
            while self.phase < -np.pi:
                self.phase += 2 * np.pi

        return output
