#!/usr/bin/env python3
"""
Paper-style DEFINED backend adapted for the SDR experiment harness.
"""

from __future__ import annotations

from paper_transformer_backend import get_runtime


def detect_symbol_with_context(context, query_symbol: complex, config):
    runtime = get_runtime("defined", config.modulation, total_symbols=31)
    return runtime.detect_symbol_with_context(context, query_symbol)


if __name__ == "__main__":
    print(
        "PaperDEFINED.py is now an importable backend module. "
        "Use it through receiver_experiment.py or the detector interface."
    )
