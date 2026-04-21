#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Headless oversampled receiver for square M-QAM experiments.

Unlike the QPSK receiver, this path does not rely on live symbol-sync output.
It captures:
  - raw IQ
  - AGC/delay processed samples
  - optional post-carrier-recovery samples

The current production validator then performs offline phase search and pilot-
based equalization on the processed stream.
"""

from __future__ import annotations

import signal
import sys
import time

from gnuradio import analog
from gnuradio import blocks
from gnuradio import digital
from gnuradio import gr
from gnuradio import soapy

from core.modulation import normalize_modulation


class ReceiveSignalMQAMHeadless(gr.top_block):
    def __init__(
        self,
        duration: float = 10.0,
        processed_output_file: str = "received_processed_mqam.bin",
        raw_iq_output_file: str = "received_iq_raw_mqam.bin",
        device_args: str = "driver=plutosdr",
        modulation: str = "16QAM",
        samp_rate: int = 1_000_000,
        symbol_rate: int = 40_000,
        center_freq: float = 915e6,
        rx_gain: float = 40.0,
        carrier_recovery: str = "none",
        costas_bw: float = 0.001,
        agc_rate: float = 5e-3,
        agc_delay_samples: int = 50_000,
    ):
        gr.top_block.__init__(self, f"{modulation} Pluto Receiver (Headless)")

        self.duration = float(duration)
        self.stop_requested = False
        self.processed_output_file = processed_output_file
        self.raw_iq_output_file = raw_iq_output_file
        self.device_args = device_args
        self.modulation = normalize_modulation(modulation)
        self.samp_rate = int(samp_rate)
        self.symbol_rate = int(symbol_rate)
        self.sps = int(self.samp_rate // self.symbol_rate)
        self.center_freq = float(center_freq)
        self.rx_gain = float(rx_gain)
        self.carrier_recovery = carrier_recovery.lower()
        self.costas_bw = float(costas_bw)

        self.soapy_source = soapy.source(self.device_args, "fc32", 1, "", "", [""], [""])
        self.soapy_source.set_sample_rate(0, self.samp_rate)
        self.soapy_source.set_bandwidth(0, self.samp_rate)
        self.soapy_source.set_gain_mode(0, False)
        self.soapy_source.set_frequency(0, self.center_freq)
        self.soapy_source.set_gain(0, min(max(self.rx_gain, 0.0), 73.0))

        self.agc = analog.agc_cc(agc_rate, 1.0, 1.0, 65536)
        self.delay = blocks.delay(gr.sizeof_gr_complex, int(agc_delay_samples))
        self.raw_sink = blocks.file_sink(gr.sizeof_gr_complex, self.raw_iq_output_file, False)
        self.raw_sink.set_unbuffered(False)
        self.processed_sink = blocks.file_sink(gr.sizeof_gr_complex, self.processed_output_file, False)
        self.processed_sink.set_unbuffered(False)

        self.connect((self.soapy_source, 0), (self.raw_sink, 0))
        self.connect((self.soapy_source, 0), (self.agc, 0))
        self.connect((self.agc, 0), (self.delay, 0))

        if self.carrier_recovery == "none":
            self.connect((self.delay, 0), (self.processed_sink, 0))
            self.carrier_block = None
        elif self.carrier_recovery == "costas":
            if self.modulation == "BPSK":
                costas_order = 2
            elif self.modulation == "QPSK":
                costas_order = 4
            else:
                raise ValueError("Costas carrier recovery is only configured here for BPSK/QPSK.")
            self.carrier_block = digital.costas_loop_cc(self.costas_bw, costas_order, False)
            self.connect((self.delay, 0), (self.carrier_block, 0))
            self.connect((self.carrier_block, 0), (self.processed_sink, 0))
        else:
            raise ValueError("carrier_recovery must be 'none' or 'costas'")

    def request_stop(self):
        self.stop_requested = True


def main() -> int:
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    processed_output_file = sys.argv[2] if len(sys.argv) > 2 else "received_processed_mqam.bin"
    raw_iq_output_file = sys.argv[3] if len(sys.argv) > 3 else "received_iq_raw_mqam.bin"
    device_args = sys.argv[4] if len(sys.argv) > 4 else "driver=plutosdr"
    modulation = sys.argv[5] if len(sys.argv) > 5 else "16QAM"
    rx_gain = float(sys.argv[6]) if len(sys.argv) > 6 else 40.0
    carrier_recovery = sys.argv[7] if len(sys.argv) > 7 else "none"
    costas_bw = float(sys.argv[8]) if len(sys.argv) > 8 else 0.001

    tb = ReceiveSignalMQAMHeadless(
        duration=duration,
        processed_output_file=processed_output_file,
        raw_iq_output_file=raw_iq_output_file,
        device_args=device_args,
        modulation=modulation,
        rx_gain=rx_gain,
        carrier_recovery=carrier_recovery,
        costas_bw=costas_bw,
    )

    def handle_signal(sig, frame):
        tb.request_stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print(f"Starting headless {tb.modulation} receiver...")
    print(f"Duration: {duration:.1f} s")
    print(f"Processed output: {processed_output_file}")
    print(f"Raw IQ output: {raw_iq_output_file}")
    print(f"Device args: {tb.device_args}")
    print(f"Carrier recovery: {tb.carrier_recovery}")
    print(f"Center frequency: {tb.center_freq/1e6:.3f} MHz")
    print(f"Sample rate: {tb.samp_rate}")
    print(f"Symbol rate: {tb.symbol_rate}")
    print(f"RX gain: {tb.rx_gain}")
    if tb.carrier_block is not None:
        print(f"Costas BW: {tb.costas_bw}")
    print()

    tb.start()
    start = time.time()
    try:
        while not tb.stop_requested and time.time() - start < duration:
            time.sleep(0.1)
    finally:
        tb.stop()
        tb.wait()
        print("Receiver stopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
