#!/usr/bin/env python3
"""
Paper-style vanilla ICL backend adapted for the SDR experiment harness.
"""

from __future__ import annotations

from paper_transformer_backend import get_runtime


def detect_symbol_with_context(context, query_symbol: complex, config):
    runtime = get_runtime("icl", config.modulation, total_symbols=31)
    return runtime.detect_symbol_with_context(context, query_symbol)


if __name__ == "__main__":
    print(
        "PaperICL.py is now an importable backend module. "
        "Use it through receiver_experiment.py or the detector interface."
    )
