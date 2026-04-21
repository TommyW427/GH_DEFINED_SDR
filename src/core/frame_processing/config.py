#!/usr/bin/env python3
"""
Frame Configuration
===================

Configuration dataclass for frame structure parameters.
"""

from dataclasses import dataclass

from core.modulation import bits_per_symbol as modulation_bits_per_symbol


@dataclass
class FrameConfig:
    """Frame structure configuration"""
    preamble_bits: int = 2000
    pilot_bits: int = 64
    data_bits: int = 64
    modulation: str = "BPSK"

    @property
    def bits_per_symbol(self) -> int:
        return modulation_bits_per_symbol(self.modulation)

    @property
    def preamble_symbols(self) -> int:
        return self.preamble_bits // self.bits_per_symbol

    @property
    def pilot_symbols(self) -> int:
        return self.pilot_bits // self.bits_per_symbol

    @property
    def data_symbols(self) -> int:
        return self.data_bits // self.bits_per_symbol
