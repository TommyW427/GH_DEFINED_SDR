#!/usr/bin/env python3
"""
Frame Synchronization
=====================

Find frame boundaries and extract frame regions (preamble, pilot, data).
"""

import numpy as np
from typing import Dict, Tuple, Optional

from .config import FrameConfig


class FrameSynchronizer:
    """
    Find frame boundaries using PILOT SEARCH (not preamble correlation).

    The proven approach from ber_analysis_pipeline.py:
    - Search for pilot position in received bits
    - Check both normal and inverted phase (180° ambiguity)
    - Use pilot position to align data extraction
    """

    def __init__(self, config: FrameConfig):
        self.config = config
        self.preamble_symbols = None

    def set_preamble(self, preamble_bits: np.ndarray):
        """
        Set known preamble sequence for correlation.

        Args:
            preamble_bits: Known preamble bit sequence
        """
        # Convert bits to symbols based on modulation
        if self.config.modulation == "BPSK":
            # BPSK: 0 → -1, 1 → +1
            self.preamble_symbols = 2 * preamble_bits.astype(np.float32) - 1
        elif self.config.modulation == "QPSK":
            # QPSK: 2 bits per symbol
            # TODO: Implement QPSK mapping
            raise NotImplementedError("QPSK not yet implemented")
        else:
            raise ValueError(f"Unknown modulation: {self.config.modulation}")

        self.preamble_symbols = self.preamble_symbols.astype(np.complex64)
        print(f"[FrameSync] Preamble set: {len(self.preamble_symbols)} symbols")

    def detect_frame_start(
        self,
        rx_symbols: np.ndarray,
        threshold_factor: float = 3.0,
        window_size: int = 100
    ) -> int:
        """
        Detect start of transmission using received signal energy.

        Cannot assume transmission starts at sample 0!

        Method: Look for sudden increase in signal power above noise floor.

        Args:
            rx_symbols: Received symbols
            threshold_factor: Multiplier above noise floor (default 3x)
            window_size: Window for power estimation

        Returns:
            Index of transmission start
        """
        print(f"\n[EnergyDetect] Detecting frame start...")

        # Compute power
        power = np.abs(rx_symbols) ** 2

        # Estimate noise floor from first samples (assume no signal yet)
        noise_floor = np.mean(power[:window_size])
        threshold = noise_floor * threshold_factor

        print(f"  Noise floor: {10*np.log10(noise_floor + 1e-12):.2f} dB")
        print(f"  Threshold: {10*np.log10(threshold + 1e-12):.2f} dB")

        # Find where power exceeds threshold
        above_threshold = power > threshold

        # Look for sustained power above threshold
        for i in range(len(above_threshold) - window_size):
            if np.mean(above_threshold[i:i + window_size]) > 0.8:  # 80% of window
                print(f"  Frame start detected at sample {i}")
                return i

        # If no clear start found, assume it starts immediately
        print(f"  ⚠ No clear frame start detected, assuming start at 0")
        return 0

    def find_pilot(
        self,
        rx_symbols: np.ndarray,
        pilot_bits: np.ndarray,
        search_start: int = 0,
        search_end: Optional[int] = None,
        stride: int = 10
    ) -> Tuple[int, bool, int]:
        """
        Find pilot position by searching for best bit match.

        This is the PROVEN working approach from ber_analysis_pipeline.py.

        Args:
            rx_symbols: Received symbols
            pilot_bits: Reference pilot bit sequence
            search_start: Start of search window
            search_end: End of search window (None = search all)
            stride: Step size for search (10 = check every 10th position)

        Returns:
            Tuple of (pilot_position, phase_inverted, pilot_errors)
        """
        print(f"\n[PilotSearch] Searching for {len(pilot_bits)}-bit pilot...")

        # Detect bits in both orientations
        rx_bits_normal = (rx_symbols.real > 0).astype(np.uint8)
        rx_bits_inverted = 1 - rx_bits_normal

        # Set search window
        if search_end is None:
            search_end = len(rx_bits_normal) - len(pilot_bits)

        # Search for best match
        best_match = {
            'pos': search_start,
            'errors': len(pilot_bits),
            'inverted': False
        }

        for pos in range(search_start, search_end, stride):
            # Try normal phase
            errs_normal = np.sum(pilot_bits != rx_bits_normal[pos:pos+len(pilot_bits)])
            if errs_normal < best_match['errors']:
                best_match = {'pos': pos, 'errors': errs_normal, 'inverted': False}

            # Try inverted phase (180° rotation)
            errs_inv = np.sum(pilot_bits != rx_bits_inverted[pos:pos+len(pilot_bits)])
            if errs_inv < best_match['errors']:
                best_match = {'pos': pos, 'errors': errs_inv, 'inverted': True}

        pilot_pos = best_match['pos']
        phase_inverted = best_match['inverted']
        pilot_errors = best_match['errors']

        print(f"  Pilot found at position: {pilot_pos}")
        print(f"  Expected position (after preamble): ~{self.config.preamble_bits}")
        print(f"  Offset from expected: {pilot_pos - self.config.preamble_bits:+d}")
        print(f"  Pilot errors: {pilot_errors}/{len(pilot_bits)}")
        print(f"  Phase inverted: {phase_inverted}")

        return pilot_pos, phase_inverted, pilot_errors

    def find_frame(self, rx_symbols: np.ndarray, threshold: float = 0.6) -> Tuple[int, float]:
        """
        Find frame start using cross-correlation with known preamble.

        Args:
            rx_symbols: Received synchronized symbols
            threshold: Normalized correlation threshold (0-1)

        Returns:
            (frame_start_index, correlation_peak_value)
        """
        if self.preamble_symbols is None:
            raise ValueError("Must call set_preamble() first")

        # Cross-correlation with known preamble
        correlation = np.correlate(rx_symbols, self.preamble_symbols, mode='valid')
        correlation_mag = np.abs(correlation)

        # Normalize correlation
        preamble_energy = np.sum(np.abs(self.preamble_symbols)**2)
        rx_energy = np.array([
            np.sum(np.abs(rx_symbols[i:i+len(self.preamble_symbols)])**2)
            for i in range(len(correlation_mag))
        ])
        correlation_norm = correlation_mag / np.sqrt(preamble_energy * rx_energy + 1e-10)

        # Find peak
        frame_start = np.argmax(correlation_norm)
        peak_value = correlation_norm[frame_start]

        print(f"[FrameSync] Frame detected:")
        print(f"  Start index: {frame_start}")
        print(f"  Correlation peak: {peak_value:.4f}")

        if peak_value < threshold:
            print(f"  WARNING: Peak below threshold ({threshold:.4f})")

        return frame_start, peak_value

    def extract_regions_from_pilot(
        self,
        rx_symbols: np.ndarray,
        pilot_position: int,
        phase_inverted: bool,
        payload_index: int = 0
    ) -> Dict[str, np.ndarray]:
        """
        Extract pilot and data regions using PILOT POSITION for alignment.

        This ensures we're comparing the correct bits!

        Frame structure: [preamble][pilot][data0][pilot][data1]...[pilot][dataN]

        Args:
            rx_symbols: Received symbols
            pilot_position: Position where FIRST pilot was found
            phase_inverted: Whether phase is inverted (from pilot search)
            payload_index: Which payload section to extract (0-indexed)

        Returns:
            Dictionary with 'pilot', 'data', phase info
        """
        pilot_len = self.config.pilot_bits
        data_len = self.config.data_bits

        # Apply phase correction if needed
        if phase_inverted:
            print(f"[Extract] Applying 180° phase correction to symbols...")
            rx_symbols = -rx_symbols

        # Calculate position of the requested [pilot][data] section
        # The pilot_position found is for the FIRST pilot (payload 0)
        # Each subsequent payload is at: first_pilot + payload_index * (pilot + data)
        section_len = pilot_len + data_len
        section_offset = payload_index * section_len

        # Extract pilot region for this payload
        pilot_start = pilot_position + section_offset
        pilot_end = pilot_start + pilot_len

        # Extract data region (immediately after this payload's pilot)
        data_start = pilot_end
        data_end = data_start + data_len

        # Verify we have enough samples
        if data_end > len(rx_symbols):
            raise ValueError(
                f"Not enough samples for payload {payload_index}: need {data_end}, have {len(rx_symbols)}"
            )

        # Extract regions
        regions = {
            'pilot': rx_symbols[pilot_start:pilot_end],
            'data': rx_symbols[data_start:data_end],
            'pilot_position': pilot_position,
            'phase_inverted': phase_inverted,
            'payload_index': payload_index,
            'indices': {
                'pilot': (pilot_start, pilot_end),
                'data': (data_start, data_end)
            }
        }

        print(f"[Extract] Regions extracted using pilot alignment (payload {payload_index}):")
        print(f"  Pilot: {len(regions['pilot'])} symbols [{pilot_start}:{pilot_end}]")
        print(f"  Data: {len(regions['data'])} symbols [{data_start}:{data_end}]")
        print(f"  Phase corrected: {phase_inverted}")

        return regions

    def extract_regions(self, rx_symbols: np.ndarray, frame_start: int, payload_index: int = 0) -> Dict[str, np.ndarray]:
        """
        Extract preamble, pilot, and data regions from frame.

        Frame structure: [preamble][pilot][data1][pilot][data2][pilot][data3]...

        Args:
            rx_symbols: Received synchronized symbols
            frame_start: Starting index of frame
            payload_index: Which payload section to extract (0-indexed)

        Returns:
            Dictionary with 'preamble', 'pilot', 'data' symbol arrays for specified payload
        """
        preamble_len = self.config.preamble_bits
        pilot_len = self.config.pilot_bits
        data_len = self.config.data_bits

        # Extract preamble (only once at the start)
        preamble_start = frame_start
        preamble_end = preamble_start + preamble_len

        # Calculate position of the requested [pilot][data] section
        # Each section is pilot_len + data_len
        section_len = pilot_len + data_len
        section_offset = preamble_end + (payload_index * section_len)

        pilot_start = section_offset
        pilot_end = pilot_start + pilot_len

        data_start = pilot_end
        data_end = data_start + data_len

        # Verify we have enough samples
        if data_end > len(rx_symbols):
            raise ValueError(
                f"Not enough samples for payload {payload_index}: need {data_end}, have {len(rx_symbols)}"
            )

        # Extract regions
        regions = {
            'preamble': rx_symbols[preamble_start:preamble_end],
            'pilot': rx_symbols[pilot_start:pilot_end],
            'data': rx_symbols[data_start:data_end],
            'frame_start': frame_start,
            'payload_index': payload_index,
            'indices': {
                'preamble': (preamble_start, preamble_end),
                'pilot': (pilot_start, pilot_end),
                'data': (data_start, data_end)
            }
        }

        print(f"[FrameSync] Payload {payload_index} regions extracted:")
        print(f"  Preamble: {len(regions['preamble'])} symbols [{preamble_start}:{preamble_end}]")
        print(f"  Pilot: {len(regions['pilot'])} symbols [{pilot_start}:{pilot_end}]")
        print(f"  Data: {len(regions['data'])} symbols [{data_start}:{data_end}]")

        return regions
