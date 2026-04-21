#!/usr/bin/env python3
"""
Symbol Timing Recovery
======================

Mueller & Muller timing recovery for symbol synchronization.
"""

import numpy as np


class MuellerMullerTimingRecovery:
    """
    Mueller & Muller Symbol Timing Recovery

    Corrects:
        - Symbol timing offset
        - Clock frequency offset (sampling rate mismatch)

    Algorithm:
        1. Interpolate input samples to recover optimal sampling points
        2. Compute timing error using Mueller & Muller TED:
           error = real(y[n]) * real(y[n-1]) - real(y[n-1]) * real(y[n])
        3. Adjust sampling phase using loop filter
        4. Output: 1 sample per symbol (decimated from sps samples)

    Uses piecewise linear interpolation for resampling.
    """

    def __init__(
        self,
        sps: int = 25,
        loop_bw: float = 0.01,
        damping: float = 1.0,
        max_deviation: float = 0.5
    ):
        """
        Initialize timing recovery

        Args:
            sps: Samples per symbol (oversampling rate)
            loop_bw: Loop bandwidth
            damping: Damping factor
            max_deviation: Maximum timing deviation (in symbols)
        """
        self.sps = float(sps)
        self.max_deviation = max_deviation

        # Compute loop filter coefficients
        denom = 1.0 + 2.0 * damping * loop_bw + loop_bw * loop_bw
        self.alpha = (4 * damping * loop_bw) / denom
        self.beta = (4 * loop_bw * loop_bw) / denom

        # State variables
        self.mu = 0.0  # Fractional timing offset [0, 1)
        self.omega = sps  # Instantaneous samples per symbol
        self.omega_mid = sps  # Nominal samples per symbol
        self.omega_lim = sps * max_deviation  # Max deviation

        # History for interpolation and TED
        self.prev_sample = 0j
        self.prev_symbol = 0j
        self.prev_prev_symbol = 0j

    def _interpolate(self, samples: np.ndarray, mu: float, idx: int) -> np.complex128:
        """
        Linear interpolation between samples

        Args:
            samples: Input sample array
            mu: Fractional delay [0, 1)
            idx: Integer sample index

        Returns:
            Interpolated sample
        """
        if idx + 1 >= len(samples):
            # Use previous sample for boundary
            return samples[idx]

        # Linear interpolation: y = y[i] + mu * (y[i+1] - y[i])
        return samples[idx] + mu * (samples[idx + 1] - samples[idx])

    def _timing_error(self, current: np.complex128, previous: np.complex128,
                      prev_prev: np.complex128) -> float:
        """
        Mueller & Muller timing error detector

        Classic M&M TED (Gardner variant):
            error = real(y[n]) * (real(y[n-1]) - real(y[n-2])) * sign

        This uses the current sample and the slope between previous samples.

        Args:
            current: Current symbol y[n]
            previous: Previous symbol y[n-1]
            prev_prev: Symbol before previous y[n-2]

        Returns:
            Timing error
        """
        # Classic Mueller & Muller TED
        # Works well for BPSK/QPSK after carrier sync
        y_n = np.real(current)
        y_n1 = np.real(previous)
        y_n2 = np.real(prev_prev)

        # Error is proportional to the cross product
        error = y_n * (y_n1 - y_n2)

        return error

    def process(self, samples: np.ndarray) -> np.ndarray:
        """
        Apply timing recovery to input samples

        Args:
            samples: Carrier-synchronized samples (sps samples per symbol)

        Returns:
            Symbol-rate samples (1 sample per symbol)
        """
        output = []
        i = 0  # Input sample index
        symbol_count = 0

        while i < len(samples) - int(np.ceil(self.omega)) - 1:
            # Interpolate to get symbol sample at fractional offset mu
            current_symbol = self._interpolate(samples, self.mu, int(i))
            output.append(current_symbol)

            # Only update timing after we have enough history
            if symbol_count >= 2:
                # Compute timing error
                error = self._timing_error(
                    current_symbol,
                    self.prev_symbol,
                    self.prev_prev_symbol
                )

                # Update sampling rate (omega) - integral path
                self.omega = self.omega + self.beta * error

                # Clamp omega to prevent runaway
                self.omega = np.clip(
                    self.omega,
                    self.omega_mid - self.omega_lim,
                    self.omega_mid + self.omega_lim
                )

                # Update fractional timing offset (mu) - proportional + integral
                self.mu = self.mu + self.omega + self.alpha * error
            else:
                # During startup, just advance by nominal omega
                self.mu = self.mu + self.omega

            # Advance to next symbol
            mu_integer = int(np.floor(self.mu))
            self.mu = self.mu - mu_integer
            i += mu_integer

            # Save history
            self.prev_prev_symbol = self.prev_symbol
            self.prev_symbol = current_symbol
            symbol_count += 1

        return np.array(output, dtype=np.complex64)
