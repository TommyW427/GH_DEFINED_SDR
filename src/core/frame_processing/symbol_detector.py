#!/usr/bin/env python3
"""
Symbol Detection
================

Detect data symbols using estimated channel.
Supports MRC, ML, and Transformer-based detection.
"""

import numpy as np
from typing import Callable

from .config import FrameConfig


class SymbolDetector:
    """
    Detect data symbols using estimated channel.
    """

    def __init__(self, config: FrameConfig):
        self.config = config

    def detect_MRC(self, data_rx: np.ndarray, channel_est: np.ndarray) -> np.ndarray:
        """
        Maximum Ratio Combining (MRC) detection for SISO.

        Equalization: y_eq = y / H_hat
        Hard decision: quantize to nearest constellation point

        Args:
            data_rx: Received data symbols (noisy)
            channel_est: Estimated channel coefficient

        Returns:
            Detected bits
        """
        print(f"[Detection] Using MRC detector")

        # Equalize
        y_eq = data_rx / (channel_est + 1e-10)

        # Hard decision for BPSK
        if self.config.modulation == "BPSK":
            # Sign of real part: -1 → 0, +1 → 1
            detected = np.sign(np.real(y_eq))
            detected_bits = ((detected + 1) / 2).astype(np.uint8)
        else:
            raise NotImplementedError(f"Detection for {self.config.modulation} not implemented")

        return detected_bits

    def detect_ML(self, data_rx: np.ndarray, channel_est: np.ndarray) -> np.ndarray:
        """
        Maximum Likelihood detection.

        Find constellation point that minimizes |y - H*x|²

        Args:
            data_rx: Received data symbols
            channel_est: Estimated channel coefficient

        Returns:
            Detected bits
        """
        print(f"[Detection] Using ML detector")

        if self.config.modulation == "BPSK":
            constellation = np.array([-1+0j, 1+0j], dtype=np.complex64)
        else:
            raise NotImplementedError()

        detected_bits = np.zeros(len(data_rx), dtype=np.uint8)

        for i, y in enumerate(data_rx):
            # Compute metric: |y - H*x|² for each constellation point
            distances = np.abs(y - channel_est * constellation)**2
            detected_bits[i] = np.argmin(distances)

        return detected_bits

    def detect_Transformer(
        self,
        data_rx: np.ndarray,
        pilot_rx: np.ndarray,
        pilot_tx: np.ndarray,
        model_func: Callable,
        use_decision_feedback: bool = True,
        **kwargs
    ) -> np.ndarray:
        """
        Transformer-based symbol detection using DEFINED.

        ====================================================================
        PLACEHOLDER FOR DEFINED DECISION FEEDBACK IN-CONTEXT DETECTION
        ====================================================================

        This is the key innovation from the ICC 2025 paper:

        Vanilla ICL:
            Prompt = [(y_pilot_1, x_pilot_1), ..., (y_pilot_k, x_pilot_k), y_data_t]
            Output = x_hat_t

        DEFINED with Decision Feedback:
            Prompt = [(y_pilot_1, x_pilot_1), ..., (y_pilot_k, x_pilot_k),
                      (y_data_{k+1}, x_hat_{k+1}), ..., (y_data_{t-1}, x_hat_{t-1}),
                      y_data_t]
            Output = x_hat_t

        The model sequentially incorporates detected symbols into the prompt
        to improve detections for subsequent symbols.

        Args:
            data_rx: Received data symbols
            pilot_rx: Received pilot symbols
            pilot_tx: Transmitted pilot symbols
            model_func: Transformer model function
            use_decision_feedback: Whether to use decision feedback (DEFINED)
            **kwargs: Additional model parameters

        Returns:
            Detected bits
        """
        print(f"[Detection] Using Transformer-based detector (DEFINED)")
        print(f"  Decision feedback: {use_decision_feedback}")

        # ====================================================================
        # TODO: Integrate DEFINED Transformer model
        # ====================================================================
        #
        # Steps:
        #   1. Construct initial ICL prompt from pilot pairs
        #   2. For each data symbol sequentially:
        #       a. Add symbol to prompt
        #       b. Run Transformer inference
        #       c. Get detection
        #       d. If using decision feedback, add (y_t, x_hat_t) to prompt
        #   3. Return all detected symbols
        #
        # This is the core DEFINED algorithm from the ICC 2025 paper.
        # ====================================================================

        if model_func is None:
            raise ValueError("Must provide model_func for Transformer detection")

        # Placeholder: call user-provided model function
        detected_bits = model_func(
            data_rx,
            pilot_rx,
            pilot_tx,
            use_decision_feedback=use_decision_feedback,
            **kwargs
        )

        return detected_bits
