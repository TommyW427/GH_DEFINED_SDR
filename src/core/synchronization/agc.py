#!/usr/bin/env python3
"""
Automatic Gain Control (AGC)
============================

Normalizes signal amplitude to a reference level using a feedback loop.
"""

import numpy as np


class AGC:
    """
    Automatic Gain Control

    Normalizes signal amplitude to a reference level using a feedback loop.

    Algorithm:
        gain[n] = gain[n-1] + rate * (reference - |x[n]|)
        y[n] = gain[n] * x[n]

    This prevents signal saturation and normalizes power for downstream blocks.
    """

    def __init__(self, rate: float = 1e-4, reference: float = 1.0, max_gain: float = 65536):
        """
        Initialize AGC

        Args:
            rate: Adaptation rate (smaller = slower, more stable)
            reference: Target amplitude
            max_gain: Maximum allowed gain (prevents runaway)
        """
        self.rate = rate
        self.reference = reference
        self.max_gain = max_gain
        self.gain = 1.0  # Initial gain

    def process(self, samples: np.ndarray) -> np.ndarray:
        """
        Apply AGC to input samples

        Args:
            samples: Complex input samples

        Returns:
            AGC-normalized samples
        """
        output = np.zeros_like(samples, dtype=np.complex64)

        for i, x in enumerate(samples):
            # Compute error: difference between reference and signal magnitude
            error = self.reference - np.abs(x)

            # Update gain using feedback
            self.gain = self.gain + self.rate * error

            # Clamp gain to prevent instability
            self.gain = np.clip(self.gain, 0.0, self.max_gain)

            # Apply gain
            output[i] = self.gain * x

        return output
