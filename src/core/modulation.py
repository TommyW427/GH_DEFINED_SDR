#!/usr/bin/env python3
"""
Shared digital modulation helpers.

These utilities centralize bit/symbol mapping so SDR validation, detector
backends, and future DEFINED integrations all use the same Gray-coded
constellation definitions.
"""

from __future__ import annotations

import math

import numpy as np


SUPPORTED_MODULATIONS = ("BPSK", "QPSK", "16QAM", "64QAM")

_GRAY_2BIT_TO_LEVEL = {
    (0, 0): -3.0,
    (0, 1): -1.0,
    (1, 1): 1.0,
    (1, 0): 3.0,
}
_LEVEL_TO_GRAY_2BIT = {int(v): np.array(k, dtype=np.uint8) for k, v in _GRAY_2BIT_TO_LEVEL.items()}

_GRAY_3BIT_TO_LEVEL = {
    (0, 0, 0): -7.0,
    (0, 0, 1): -5.0,
    (0, 1, 1): -3.0,
    (0, 1, 0): -1.0,
    (1, 1, 0): 1.0,
    (1, 1, 1): 3.0,
    (1, 0, 1): 5.0,
    (1, 0, 0): 7.0,
}
_LEVEL_TO_GRAY_3BIT = {int(v): np.array(k, dtype=np.uint8) for k, v in _GRAY_3BIT_TO_LEVEL.items()}


def normalize_modulation(modulation: str) -> str:
    name = modulation.upper()
    if name not in SUPPORTED_MODULATIONS:
        raise ValueError(f"Unsupported modulation: {modulation}")
    return name


def bits_per_symbol(modulation: str) -> int:
    name = normalize_modulation(modulation)
    if name == "BPSK":
        return 1
    if name == "QPSK":
        return 2
    if name == "16QAM":
        return 4
    return 6


def average_symbol_energy(modulation: str) -> float:
    name = normalize_modulation(modulation)
    if name == "BPSK":
        return 1.0
    if name == "QPSK":
        return 1.0
    if name == "16QAM":
        return 10.0
    return 42.0


def normalization(modulation: str) -> float:
    return math.sqrt(average_symbol_energy(modulation))


def _bits_to_16qam_symbol(bits: np.ndarray) -> complex:
    i_level = _GRAY_2BIT_TO_LEVEL[(int(bits[0]), int(bits[1]))]
    q_level = _GRAY_2BIT_TO_LEVEL[(int(bits[2]), int(bits[3]))]
    return complex(i_level, q_level) / normalization("16QAM")


def _bits_to_64qam_symbol(bits: np.ndarray) -> complex:
    i_level = _GRAY_3BIT_TO_LEVEL[(int(bits[0]), int(bits[1]), int(bits[2]))]
    q_level = _GRAY_3BIT_TO_LEVEL[(int(bits[3]), int(bits[4]), int(bits[5]))]
    return complex(i_level, q_level) / normalization("64QAM")


def symbol_from_bits(bits: np.ndarray, modulation: str) -> complex:
    name = normalize_modulation(modulation)
    if name == "BPSK":
        return complex(2 * int(bits[0]) - 1, 0.0)
    if name == "QPSK":
        b0, b1 = int(bits[0]), int(bits[1])
        if b0 == 0 and b1 == 0:
            return 0.70710678 + 0.70710678j
        if b0 == 0 and b1 == 1:
            return -0.70710678 + 0.70710678j
        if b0 == 1 and b1 == 0:
            return 0.70710678 - 0.70710678j
        return -0.70710678 - 0.70710678j
    if name == "16QAM":
        return _bits_to_16qam_symbol(bits)
    return _bits_to_64qam_symbol(bits)


def bits_to_symbols(bits: np.ndarray, modulation: str) -> np.ndarray:
    name = normalize_modulation(modulation)
    k = bits_per_symbol(name)
    if len(bits) % k != 0:
        raise ValueError(f"{name} bit arrays must have length divisible by {k}.")
    if name == "BPSK":
        return (2 * bits.astype(np.float32) - 1).astype(np.complex64)
    groups = bits.reshape(-1, k)
    out = np.empty(len(groups), dtype=np.complex64)
    for idx, group in enumerate(groups):
        out[idx] = symbol_from_bits(group, name)
    return out


def _nearest_level(values: np.ndarray, levels: np.ndarray) -> np.ndarray:
    distances = np.abs(values[:, None] - levels[None, :])
    return levels[np.argmin(distances, axis=1)]


def symbols_to_bits(symbols: np.ndarray, modulation: str) -> np.ndarray:
    name = normalize_modulation(modulation)
    if name == "BPSK":
        return (np.real(symbols) >= 0).astype(np.uint8)
    if name == "QPSK":
        bits = np.zeros(len(symbols) * 2, dtype=np.uint8)
        q1 = (symbols.real >= 0) & (symbols.imag >= 0)
        q2 = (symbols.real < 0) & (symbols.imag >= 0)
        q3 = (symbols.real < 0) & (symbols.imag < 0)
        q4 = (symbols.real >= 0) & (symbols.imag < 0)
        bits[0::2][q1] = 0
        bits[1::2][q1] = 0
        bits[0::2][q2] = 0
        bits[1::2][q2] = 1
        bits[0::2][q3] = 1
        bits[1::2][q3] = 1
        bits[0::2][q4] = 1
        bits[1::2][q4] = 0
        return bits

    scale = normalization(name)
    rescaled = np.asarray(symbols, dtype=np.complex64) * scale
    if name == "16QAM":
        levels = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float32)
        i_levels = _nearest_level(rescaled.real, levels).astype(np.int32)
        q_levels = _nearest_level(rescaled.imag, levels).astype(np.int32)
        bits = np.empty(len(symbols) * 4, dtype=np.uint8)
        bits[0::4] = [_LEVEL_TO_GRAY_2BIT[level][0] for level in i_levels]
        bits[1::4] = [_LEVEL_TO_GRAY_2BIT[level][1] for level in i_levels]
        bits[2::4] = [_LEVEL_TO_GRAY_2BIT[level][0] for level in q_levels]
        bits[3::4] = [_LEVEL_TO_GRAY_2BIT[level][1] for level in q_levels]
        return bits

    levels = np.array([-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=np.float32)
    i_levels = _nearest_level(rescaled.real, levels).astype(np.int32)
    q_levels = _nearest_level(rescaled.imag, levels).astype(np.int32)
    bits = np.empty(len(symbols) * 6, dtype=np.uint8)
    bits[0::6] = [_LEVEL_TO_GRAY_3BIT[level][0] for level in i_levels]
    bits[1::6] = [_LEVEL_TO_GRAY_3BIT[level][1] for level in i_levels]
    bits[2::6] = [_LEVEL_TO_GRAY_3BIT[level][2] for level in i_levels]
    bits[3::6] = [_LEVEL_TO_GRAY_3BIT[level][0] for level in q_levels]
    bits[4::6] = [_LEVEL_TO_GRAY_3BIT[level][1] for level in q_levels]
    bits[5::6] = [_LEVEL_TO_GRAY_3BIT[level][2] for level in q_levels]
    return bits


def constellation_points(modulation: str) -> list[complex]:
    name = normalize_modulation(modulation)
    k = bits_per_symbol(name)
    points: list[complex] = []
    for index in range(1 << k):
        bits = np.array([(index >> shift) & 1 for shift in range(k - 1, -1, -1)], dtype=np.uint8)
        points.append(symbol_from_bits(bits, name))
    return points
