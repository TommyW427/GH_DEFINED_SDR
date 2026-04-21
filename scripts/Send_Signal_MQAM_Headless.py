#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Headless square M-QAM Pluto transmitter.

Supports QPSK, 16QAM, and 64QAM with shared Gray-coded constellations from
`core.modulation`.
"""

from __future__ import annotations

import signal
import sys
import time

import pmt
from gnuradio import blocks
from gnuradio import digital
from gnuradio import gr
from gnuradio import soapy

from core.modulation import bits_per_symbol, constellation_points, normalize_modulation


class SendSignalMQAMHeadless(gr.top_block):
    def __init__(
        self,
        bit_file: str,
        modulation: str,
        duration: float = 5.0,
        repeat: bool = True,
        device_args: str = "driver=plutosdr",
        samp_rate: int = 1_000_000,
        symbol_rate: int = 40_000,
        center_freq: float = 915e6,
        tx_gain: float = 50.0,
        tx_scale: float = 0.4,
    ):
        gr.top_block.__init__(self, f"{modulation} Pluto Transmitter (Headless)")

        self.bit_file = bit_file
        self.modulation = normalize_modulation(modulation)
        self.duration = float(duration)
        self.repeat = bool(repeat)
        self.stop_requested = False
        self.device_args = device_args
        self.samp_rate = int(samp_rate)
        self.symbol_rate = int(symbol_rate)
        self.sps = int(self.samp_rate // self.symbol_rate)
        self.center_freq = float(center_freq)
        self.tx_gain = float(tx_gain)
        self.tx_scale = float(tx_scale)

        self.soapy_sink = soapy.sink(self.device_args, "fc32", 1, "", "", [""], [""])
        self.soapy_sink.set_sample_rate(0, self.samp_rate)
        self.soapy_sink.set_bandwidth(0, self.samp_rate)
        self.soapy_sink.set_frequency(0, self.center_freq)
        self.soapy_sink.set_gain(0, min(max(self.tx_gain, 0.0), 89.0))

        self.file_source = blocks.file_source(gr.sizeof_char, self.bit_file, self.repeat, 0, 0)
        self.file_source.set_begin_tag(pmt.PMT_NIL)
        self.pack_bits = blocks.pack_k_bits_bb(bits_per_symbol(self.modulation))
        self.map_symbols = digital.chunks_to_symbols_bc(constellation_points(self.modulation))
        self.repeat_symbols = blocks.repeat(gr.sizeof_gr_complex, self.sps)
        self.scale = blocks.multiply_const_cc(self.tx_scale)

        self.connect((self.file_source, 0), (self.pack_bits, 0))
        self.connect((self.pack_bits, 0), (self.map_symbols, 0))
        self.connect((self.map_symbols, 0), (self.repeat_symbols, 0))
        self.connect((self.repeat_symbols, 0), (self.scale, 0))
        self.connect((self.scale, 0), (self.soapy_sink, 0))

    def request_stop(self):
        self.stop_requested = True


def main() -> int:
    bit_file = sys.argv[1] if len(sys.argv) > 1 else "frames/test_frame_16qam.bin"
    modulation = sys.argv[2] if len(sys.argv) > 2 else "16QAM"
    duration = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
    repeat = sys.argv[4].lower() not in {"0", "false", "no"} if len(sys.argv) > 4 else True
    device_args = sys.argv[5] if len(sys.argv) > 5 else "driver=plutosdr"
    tx_gain = float(sys.argv[6]) if len(sys.argv) > 6 else 50.0
    tx_scale = float(sys.argv[7]) if len(sys.argv) > 7 else 0.4

    tb = SendSignalMQAMHeadless(
        bit_file=bit_file,
        modulation=modulation,
        duration=duration,
        repeat=repeat,
        device_args=device_args,
        tx_gain=tx_gain,
        tx_scale=tx_scale,
    )

    def handle_signal(sig, frame):
        tb.request_stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print(f"Starting headless {tb.modulation} transmitter...")
    print(f"Input file: {bit_file}")
    print(f"Duration: {duration:.1f} s")
    print(f"Repeat: {repeat}")
    print(f"Device args: {tb.device_args}")
    print(f"Center frequency: {tb.center_freq/1e6:.3f} MHz")
    print(f"Sample rate: {tb.samp_rate}")
    print(f"Symbol rate: {tb.symbol_rate}")
    print(f"TX gain: {tb.tx_gain}")
    print(f"TX scale: {tb.tx_scale}")
    print()

    tb.start()
    start = time.time()
    try:
        while not tb.stop_requested and time.time() - start < duration:
            time.sleep(0.1)
    finally:
        tb.stop()
        tb.wait()
        print("Transmitter stopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
