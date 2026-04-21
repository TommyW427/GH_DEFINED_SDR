#!/usr/bin/env python3
"""
Detector abstractions for paper-style symbol detection experiments.

This module keeps the SDR front-end separate from the actual detector logic.
It assumes synchronization/alignment has already happened and provides:
  - abstract prompt/context structures
  - coherent MMSE-style baselines
  - MMSE with decision feedback
  - an adapter interface for a future DEFINED backend
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from core.modulation import bits_to_symbols, symbols_to_bits
from .config import FrameConfig


PromptSource = Literal["pilot", "feedback"]


@dataclass
class ContextPair:
    received: complex
    transmitted: complex
    source: PromptSource


@dataclass
class DetectionStep:
    index: int
    received: complex
    detected_symbol: complex
    detected_bits: np.ndarray
    context_length: int


@dataclass
class DetectionRunResult:
    detector_name: str
    config: FrameConfig
    steps: list[DetectionStep]
    context_pairs: list[ContextPair]
    detected_bits: np.ndarray


@dataclass
class DetectorFrameInputs:
    config: FrameConfig
    pilot_rx: np.ndarray
    pilot_tx_bits: np.ndarray
    data_rx: np.ndarray

    @property
    def pilot_tx_symbols(self) -> np.ndarray:
        return bits_to_symbols(self.pilot_tx_bits, self.config.modulation)


class DecisionFeedbackPromptBuilder:
    """
    Build the sequence of clean pilot pairs plus decision-feedback pairs.
    """

    def __init__(self, frame_inputs: DetectorFrameInputs):
        self.frame_inputs = frame_inputs

    def initial_context(self) -> list[ContextPair]:
        return [
            ContextPair(received=complex(rx), transmitted=complex(tx), source="pilot")
            for rx, tx in zip(self.frame_inputs.pilot_rx, self.frame_inputs.pilot_tx_symbols)
        ]

    def append_feedback(
        self,
        context: list[ContextPair],
        received_symbol: complex,
        detected_symbol: complex,
    ) -> list[ContextPair]:
        updated = list(context)
        updated.append(
            ContextPair(
                received=complex(received_symbol),
                transmitted=complex(detected_symbol),
                source="feedback",
            )
        )
        return updated


class AbstractDetectorBackend(ABC):
    def __init__(self, config: FrameConfig, name: str):
        self.config = config
        self.name = name

    @abstractmethod
    def detect(self, frame_inputs: DetectorFrameInputs) -> DetectionRunResult:
        raise NotImplementedError


class CoherentMMSEDetector(AbstractDetectorBackend):
    """
    Classical coherent baseline: estimate a single block-fading channel from pilots
    and detect all subsequent symbols with the fixed estimate.
    """

    def __init__(self, config: FrameConfig):
        super().__init__(config, "mmse")

    def estimate_channel(self, pilot_rx: np.ndarray, pilot_tx: np.ndarray) -> complex:
        h_ls = np.vdot(pilot_tx, pilot_rx) / (np.vdot(pilot_tx, pilot_tx) + 1e-12)
        residual = pilot_rx - h_ls * pilot_tx
        noise_var = float(np.mean(np.abs(residual) ** 2))
        xhy = np.vdot(pilot_tx, pilot_rx)
        xhx = np.vdot(pilot_tx, pilot_tx)
        return complex(xhy / (xhx + noise_var))

    def detect(self, frame_inputs: DetectorFrameInputs) -> DetectionRunResult:
        pilot_tx = frame_inputs.pilot_tx_symbols
        h_hat = self.estimate_channel(frame_inputs.pilot_rx, pilot_tx)
        y_eq = frame_inputs.data_rx / (h_hat + 1e-12)
        bits = symbols_to_bits(y_eq, self.config.modulation)

        bits_per_symbol = self.config.bits_per_symbol
        steps = []
        for idx, symbol in enumerate(y_eq):
            step_bits = bits[idx * bits_per_symbol:(idx + 1) * bits_per_symbol]
            hard_symbol = bits_to_symbols(step_bits, self.config.modulation)[0]
            steps.append(
                DetectionStep(
                    index=idx,
                    received=complex(frame_inputs.data_rx[idx]),
                    detected_symbol=complex(hard_symbol),
                    detected_bits=step_bits.copy(),
                    context_length=len(pilot_tx),
                )
            )

        builder = DecisionFeedbackPromptBuilder(frame_inputs)
        return DetectionRunResult(
            detector_name=self.name,
            config=self.config,
            steps=steps,
            context_pairs=builder.initial_context(),
            detected_bits=bits,
        )


class MMSEDecisionFeedbackDetector(CoherentMMSEDetector):
    """
    MMSE-DF paper-style baseline: sequentially treat previous decisions as noisy
    pilot pairs and re-estimate the block-fading channel before each decision.
    """

    def __init__(self, config: FrameConfig):
        super().__init__(config)
        self.name = "mmse_df"

    def detect(self, frame_inputs: DetectorFrameInputs) -> DetectionRunResult:
        builder = DecisionFeedbackPromptBuilder(frame_inputs)
        context = builder.initial_context()
        steps: list[DetectionStep] = []
        detected_bits = []

        for idx, rx_symbol in enumerate(frame_inputs.data_rx):
            ctx_rx = np.array([pair.received for pair in context], dtype=np.complex64)
            ctx_tx = np.array([pair.transmitted for pair in context], dtype=np.complex64)
            h_hat = self.estimate_channel(ctx_rx, ctx_tx)
            y_eq = np.array([rx_symbol / (h_hat + 1e-12)], dtype=np.complex64)
            bits = symbols_to_bits(y_eq, self.config.modulation)
            detected_symbol = bits_to_symbols(bits, self.config.modulation)[0]
            detected_bits.append(bits)
            steps.append(
                DetectionStep(
                    index=idx,
                    received=complex(rx_symbol),
                    detected_symbol=complex(detected_symbol),
                    detected_bits=bits.copy(),
                    context_length=len(context),
                )
            )
            context = builder.append_feedback(context, complex(rx_symbol), complex(detected_symbol))

        return DetectionRunResult(
            detector_name=self.name,
            config=self.config,
            steps=steps,
            context_pairs=context,
            detected_bits=np.concatenate(detected_bits) if detected_bits else np.array([], dtype=np.uint8),
        )


class DefinedDetectorAdapter(AbstractDetectorBackend):
    """
    Abstract adapter for a future DEFINED backend.

    The backend is assumed to accept the paper-style prompt structure:
      clean pilot pairs + optional decision-feedback pairs + current y_t query
    and return a detected symbol or label for each step.
    """

    def __init__(
        self,
        config: FrameConfig,
        model_func: Callable[[list[ContextPair], complex, FrameConfig], complex | np.ndarray],
        detector_name: str = "defined",
        use_decision_feedback: bool = True,
    ):
        super().__init__(config, detector_name)
        self.model_func = model_func
        self.use_decision_feedback = use_decision_feedback

    def detect(self, frame_inputs: DetectorFrameInputs) -> DetectionRunResult:
        builder = DecisionFeedbackPromptBuilder(frame_inputs)
        context = builder.initial_context()
        steps: list[DetectionStep] = []
        detected_bits = []

        for idx, rx_symbol in enumerate(frame_inputs.data_rx):
            raw_output = self.model_func(context, complex(rx_symbol), self.config)
            if isinstance(raw_output, np.ndarray):
                step_bits = raw_output.astype(np.uint8)
                detected_symbol = bits_to_symbols(step_bits, self.config.modulation)[0]
            else:
                detected_symbol = complex(raw_output)
                step_bits = symbols_to_bits(np.array([detected_symbol], dtype=np.complex64), self.config.modulation)

            detected_bits.append(step_bits)
            steps.append(
                DetectionStep(
                    index=idx,
                    received=complex(rx_symbol),
                    detected_symbol=complex(detected_symbol),
                    detected_bits=step_bits.copy(),
                    context_length=len(context),
                )
            )
            if self.use_decision_feedback:
                context = builder.append_feedback(context, complex(rx_symbol), complex(detected_symbol))

        return DetectionRunResult(
            detector_name=self.name,
            config=self.config,
            steps=steps,
            context_pairs=context,
            detected_bits=np.concatenate(detected_bits) if detected_bits else np.array([], dtype=np.uint8),
        )
