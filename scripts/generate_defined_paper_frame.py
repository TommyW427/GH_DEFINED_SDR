#!/usr/bin/env python3
"""
Generate a paper-style SDR frame for DEFINED replication.

Paper-aligned structure:
  [long sync preamble] + [k pilot symbols] + [T-k data symbols]

The SDR preamble remains longer than the paper's abstract frame because we use
it only for synchronization in the real radios. The paper-style portion begins
after the preamble and has total length T symbols.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from core.modulation import bits_per_symbol, normalize_modulation


def _corner_symbol_patterns(modulation: str) -> np.ndarray:
    modulation = normalize_modulation(modulation)
    if modulation == "BPSK":
        return np.array([[0], [1]], dtype=np.uint8)
    if modulation == "QPSK":
        return np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)
    if modulation == "16QAM":
        return np.array(
            [
                [0, 0, 0, 0],  # (-3, -3)
                [0, 0, 1, 0],  # (-3, +3)
                [1, 0, 0, 0],  # (+3, -3)
                [1, 0, 1, 0],  # (+3, +3)
            ],
            dtype=np.uint8,
        )
    return np.array(
        [
            [0, 0, 0, 0, 0, 0],  # (-7, -7)
            [0, 0, 0, 1, 0, 0],  # (-7, +7)
            [1, 0, 0, 0, 0, 0],  # (+7, -7)
            [1, 0, 0, 1, 0, 0],  # (+7, +7)
        ],
        dtype=np.uint8,
    )


def build_sync_preamble(modulation: str, preamble_symbols: int, sync_tail_symbols: int = 64) -> np.ndarray:
    bps = bits_per_symbol(modulation)
    if preamble_symbols <= 0:
        return np.empty(0, dtype=np.uint8)

    # The paper-style sweep needs an unambiguous frame boundary, but the SDR
    # front-end also needs an easy-to-lock preamble. Use a hybrid:
    #   1. long structured section that repeatedly exercises the constellation
    #   2. short pseudo-random tail that sharpens the end-of-preamble boundary
    tail_symbols = max(0, min(int(sync_tail_symbols), preamble_symbols // 4, preamble_symbols))
    body_symbols = preamble_symbols - tail_symbols

    patterns = _corner_symbol_patterns(modulation)

    if body_symbols > 0:
        repeats = (body_symbols + len(patterns) - 1) // len(patterns)
        body = np.tile(patterns, (repeats, 1))[:body_symbols].reshape(-1)
    else:
        body = np.empty(0, dtype=np.uint8)

    if tail_symbols > 0:
        rng = np.random.default_rng(7)
        tail_indices = rng.integers(0, len(patterns), size=tail_symbols)
        tail = patterns[tail_indices].reshape(-1)
        return np.concatenate([body, tail]).astype(np.uint8)
    return body.astype(np.uint8)


def generate_defined_paper_frame(
    modulation: str,
    pilot_symbols: int,
    total_symbols: int = 31,
    preamble_symbols: int = 6000,
    sync_tail_symbols: int = 64,
    symbol_rate: int = 40000,
    seed: int = 42,
    frame_output: str | Path | None = None,
    payload_output: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    modulation = normalize_modulation(modulation)
    if pilot_symbols <= 0:
        raise ValueError("pilot_symbols must be positive.")
    if pilot_symbols >= total_symbols:
        raise ValueError("pilot_symbols must be smaller than total_symbols.")

    bps = bits_per_symbol(modulation)
    data_symbols = total_symbols - pilot_symbols
    modulation_tag = modulation.lower()

    frame_path = Path(frame_output or f"frames/defined_paper_{modulation_tag}_k{pilot_symbols:02d}.bin")
    payload_path = Path(payload_output or f"payloads_{modulation_tag}/defined_paper_payload_k{pilot_symbols:02d}.bin")
    metadata_path = frame_path.with_suffix(".json")
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed + pilot_symbols)
    preamble_bits = build_sync_preamble(modulation, preamble_symbols, sync_tail_symbols=sync_tail_symbols)
    pilot_bits = rng.integers(0, 2, size=pilot_symbols * bps, dtype=np.uint8)
    data_bits = rng.integers(0, 2, size=data_symbols * bps, dtype=np.uint8)

    data_bits.tofile(payload_path)
    frame_bits = np.concatenate([preamble_bits, pilot_bits, data_bits]).astype(np.uint8)
    frame_bits.tofile(frame_path)

    metadata = {
        "modulation": modulation,
        "bits_per_symbol": bps,
        "preamble_symbols": preamble_symbols,
        "sync_tail_symbols": sync_tail_symbols,
        "pilot_symbols": pilot_symbols,
        "data_symbols": data_symbols,
        "paper_total_symbols": total_symbols,
        "paper_payload_symbols": data_symbols,
        "paper_pilot_symbols": pilot_symbols,
        "preamble_bits": preamble_symbols * bps,
        "pilot_bits": pilot_symbols * bps,
        "data_bits": data_symbols * bps,
        "num_payloads": 1,
        "payload_files": [str(payload_path)],
        "frame_length_bits": int(len(frame_bits)),
        "frame_length_symbols": int(len(frame_bits) // bps),
        "symbol_rate": symbol_rate,
        "frame_duration_ms": (len(frame_bits) / bps) / symbol_rate * 1000.0,
        "paper_frame_duration_ms": total_symbols / symbol_rate * 1000.0,
        "note": "DEFINED paper-style frame: structured SDR sync preamble plus short unique tail, followed by one block of k pilots and T-k data symbols",
    }
    if bps == 2:
        metadata["preamble_dibits"] = preamble_symbols
        metadata["pilot_dibits"] = pilot_symbols
        metadata["data_dibits"] = data_symbols

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return frame_path, metadata_path, payload_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a DEFINED paper-style SDR frame.")
    parser.add_argument("--modulation", required=True, choices=["BPSK", "QPSK", "16QAM", "64QAM"])
    parser.add_argument("--pilot-symbols", required=True, type=int)
    parser.add_argument("--total-symbols", type=int, default=31)
    parser.add_argument("--preamble-symbols", type=int, default=6000)
    parser.add_argument("--sync-tail-symbols", type=int, default=64)
    parser.add_argument("--symbol-rate", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame-output", default=None)
    parser.add_argument("--payload-output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame_path, metadata_path, payload_path = generate_defined_paper_frame(
        modulation=args.modulation,
        pilot_symbols=args.pilot_symbols,
        total_symbols=args.total_symbols,
        preamble_symbols=args.preamble_symbols,
        sync_tail_symbols=args.sync_tail_symbols,
        symbol_rate=args.symbol_rate,
        seed=args.seed,
        frame_output=args.frame_output,
        payload_output=args.payload_output,
    )

    print("=" * 72)
    print("DEFINED PAPER FRAME")
    print("=" * 72)
    print(f"Frame:    {frame_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Payload:  {payload_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
