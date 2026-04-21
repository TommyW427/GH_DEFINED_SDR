#!/usr/bin/env python3
"""
Unified experiment runner for aligned SDR payloads.

This module bridges:
  post-Costas alignment/phase selection
  -> detector backend interface
  -> payload-level and aggregate BER reporting.

It keeps the detector interface abstract so a future DEFINED backend can be
plugged in without changing the SDR alignment path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import FrameConfig
from .detector_interfaces import (
    DetectorFrameInputs,
    CoherentMMSEDetector,
    MMSEDecisionFeedbackDetector,
    DefinedDetectorAdapter,
    DetectionRunResult,
)


@dataclass
class PayloadCase:
    payload_index: int
    expected_payload_index: int
    frame_inputs: DetectorFrameInputs
    expected_bits: np.ndarray
    pilot_score: float
    pilot_ber: float
    pilot_drift: int


@dataclass
class PayloadDetectionResult:
    payload_index: int
    expected_payload_index: int
    detector_name: str
    bit_errors: int
    bits: int
    ber: float
    pilot_score: float
    pilot_ber: float
    pilot_drift: int
    run_result: DetectionRunResult


@dataclass
class DetectorExperimentSummary:
    detector_name: str
    payload_results: list[PayloadDetectionResult]
    total_errors: int
    total_bits: int
    overall_ber: float


def build_payload_cases_from_phase_result(
    phase_result: dict,
    decimated_symbols: np.ndarray,
    metadata: dict,
    tx_bits: np.ndarray,
    expected_payload_bits: list[np.ndarray],
) -> list[PayloadCase]:
    """
    Construct per-payload DetectorFrameInputs from a best-phase post-Costas result.
    """
    config = FrameConfig(
        preamble_bits=metadata["preamble_bits"],
        pilot_bits=metadata["pilot_bits"],
        data_bits=metadata["data_bits"],
        modulation=metadata.get("modulation", "QPSK"),
    )
    pilot_bits = tx_bits[metadata["preamble_bits"]:metadata["preamble_bits"] + metadata["pilot_bits"]]
    pilot_symbols = metadata.get("pilot_symbols", metadata.get("pilot_dibits", config.pilot_symbols))
    data_symbols = metadata.get("data_symbols", metadata.get("data_dibits", config.data_symbols))

    cases = []
    for payload_info in phase_result["payload_results"]:
        pilot_pos = payload_info["pilot_pos"]
        rx_pilot = decimated_symbols[pilot_pos:pilot_pos + pilot_symbols]
        rx_payload = decimated_symbols[pilot_pos + pilot_symbols:pilot_pos + pilot_symbols + data_symbols]
        if len(rx_pilot) != pilot_symbols or len(rx_payload) != data_symbols:
            continue

        expected_idx = payload_info["expected_payload_index"]
        cases.append(
            PayloadCase(
                payload_index=payload_info["payload_index"],
                expected_payload_index=expected_idx,
                frame_inputs=DetectorFrameInputs(
                    config=config,
                    pilot_rx=rx_pilot.astype(np.complex64),
                    pilot_tx_bits=pilot_bits.astype(np.uint8),
                    data_rx=rx_payload.astype(np.complex64),
                ),
                expected_bits=expected_payload_bits[expected_idx].astype(np.uint8),
                pilot_score=float(payload_info["pilot_score"]),
                pilot_ber=float(payload_info["pilot_ber"]),
                pilot_drift=int(payload_info["pilot_drift"]),
            )
        )
    return cases


def run_detector_on_cases(
    detector,
    payload_cases: list[PayloadCase],
) -> DetectorExperimentSummary:
    payload_results: list[PayloadDetectionResult] = []
    total_errors = 0
    total_bits = 0

    for case in payload_cases:
        run_result = detector.detect(case.frame_inputs)
        detected_bits = run_result.detected_bits[: len(case.expected_bits)]
        bit_errors = int(np.sum(detected_bits != case.expected_bits))
        bits = int(len(case.expected_bits))
        ber = bit_errors / bits if bits else 1.0
        total_errors += bit_errors
        total_bits += bits
        payload_results.append(
            PayloadDetectionResult(
                payload_index=case.payload_index,
                expected_payload_index=case.expected_payload_index,
                detector_name=detector.name,
                bit_errors=bit_errors,
                bits=bits,
                ber=ber,
                pilot_score=case.pilot_score,
                pilot_ber=case.pilot_ber,
                pilot_drift=case.pilot_drift,
                run_result=run_result,
            )
        )

    return DetectorExperimentSummary(
        detector_name=detector.name,
        payload_results=payload_results,
        total_errors=total_errors,
        total_bits=total_bits,
        overall_ber=(total_errors / total_bits) if total_bits else 1.0,
    )


def make_detector(
    detector_name: str,
    config: FrameConfig,
    defined_model_func: Callable | None = None,
    icl_model_func: Callable | None = None,
):
    name = detector_name.lower()
    if name == "mmse":
        return CoherentMMSEDetector(config)
    if name == "mmse_df":
        return MMSEDecisionFeedbackDetector(config)
    if name == "icl":
        if icl_model_func is None:
            raise ValueError("icl_model_func is required for detector='icl'")
        return DefinedDetectorAdapter(
            config,
            model_func=icl_model_func,
            detector_name="icl",
            use_decision_feedback=False,
        )
    if name == "defined":
        if defined_model_func is None:
            raise ValueError("defined_model_func is required for detector='defined'")
        return DefinedDetectorAdapter(
            config,
            model_func=defined_model_func,
            detector_name="defined",
            use_decision_feedback=True,
        )
    raise ValueError(f"Unsupported detector '{detector_name}'")
