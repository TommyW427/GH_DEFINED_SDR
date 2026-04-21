#!/usr/bin/env python3
"""
QPSK BER/SER Validation
=======================

Advanced validation for received QPSK frames with multi-payload support.
Follows the same proven architecture as ber_validator.py for BPSK.
"""

import numpy as np
import json
from pathlib import Path


def qpsk_symbol_to_bits(symbol, phase_correction=0):
    """
    Demodulate a single QPSK symbol to 2 bits using ML decision.

    Args:
        symbol: Complex QPSK symbol
        phase_correction: Phase rotation to apply (radians)

    Gray-coded QPSK:
    - Q1 (+I,+Q) → 00
    - Q2 (-I,+Q) → 01
    - Q3 (-I,-Q) → 11
    - Q4 (+I,-Q) → 10
    """
    # Apply phase correction
    corrected = symbol * np.exp(-1j * phase_correction)

    if corrected.real > 0 and corrected.imag > 0:  # Q1
        return np.array([0, 0], dtype=np.uint8)
    elif corrected.real < 0 and corrected.imag > 0:  # Q2
        return np.array([0, 1], dtype=np.uint8)
    elif corrected.real < 0 and corrected.imag < 0:  # Q3
        return np.array([1, 1], dtype=np.uint8)
    else:  # Q4 (+I,-Q)
        return np.array([1, 0], dtype=np.uint8)


def bits_to_qpsk_symbols(bits):
    """
    Convert bit array to QPSK symbols (for pilot reference).

    Args:
        bits: Array of bits (length must be even)

    Returns:
        Complex array of QPSK symbols (normalized to unit energy)
    """
    if len(bits) % 2 != 0:
        raise ValueError("Bit array length must be even for QPSK")

    bit_pairs = bits.reshape(-1, 2)
    symbols = np.zeros(len(bit_pairs), dtype=np.complex64)

    for i, (b0, b1) in enumerate(bit_pairs):
        if b0 == 0 and b1 == 0:  # 00
            symbols[i] = 0.707 + 0.707j
        elif b0 == 0 and b1 == 1:  # 01
            symbols[i] = -0.707 + 0.707j
        elif b0 == 1 and b1 == 0:  # 10
            symbols[i] = 0.707 - 0.707j
        else:  # 11
            symbols[i] = -0.707 - 0.707j

    return symbols


def validate_frame_qpsk(tx_frame_file, rx_symbols_file, sps=1, metadata_file=None):
    """
    Validate BER and SER for received QPSK frame with full payload analysis.

    Follows the same proven approach as BPSK validate_frame().

    Args:
        tx_frame_file: Transmitted frame bits (e.g., 'frames/test_frame_qpsk.bin')
        rx_symbols_file: Received symbols ('received_symbols_qpsk.bin')
        sps: Additional decimation (default 1)
        metadata_file: Optional JSON metadata file

    Returns:
        dict with BER, SER, per-payload analysis
    """
    # Load metadata
    if metadata_file is None:
        metadata_file = 'frames/test_frame_qpsk.json'

    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        PREAMBLE_BITS = metadata['preamble_bits']
        PILOT_BITS = metadata['pilot_bits']
        DATA_BITS = metadata['data_bits']
        NUM_PAYLOADS = metadata['num_payloads']
        PREAMBLE_DIBITS = metadata['preamble_dibits']
        PILOT_DIBITS = metadata['pilot_dibits']
        DATA_DIBITS = metadata['data_dibits']
    except Exception as e:
        print(f"Warning: Could not load metadata from {metadata_file}: {e}")
        print("Using default QPSK values...")
        PREAMBLE_DIBITS = 6000
        PREAMBLE_BITS = 12000
        PILOT_DIBITS = 64
        PILOT_BITS = 128
        DATA_DIBITS = 64
        DATA_BITS = 128
        NUM_PAYLOADS = 10

    # Load transmitted bits
    tx_bits = np.fromfile(tx_frame_file, dtype=np.uint8)

    # Load received symbols
    rx_symbols = np.fromfile(rx_symbols_file, dtype=np.complex64)

    # Decimate if needed
    rx_symbols_decimated = rx_symbols if sps == 1 else rx_symbols[::sps]

    # Frame structure (in symbols)
    FRAME_LEN = PILOT_DIBITS + DATA_DIBITS  # symbols per payload

    print("="*70)
    print("QPSK FRAME VALIDATION")
    print("="*70)
    print(f"TX bits: {len(tx_bits)}")
    print(f"RX symbols (raw): {len(rx_symbols)}")
    print(f"RX symbols (decimated): {len(rx_symbols_decimated)}")
    print(f"SPS: {sps}")
    print()
    print(f"Frame structure:")
    print(f"  Preamble: {PREAMBLE_DIBITS} symbols ({PREAMBLE_BITS} bits)")
    print(f"  Pilot: {PILOT_DIBITS} symbols ({PILOT_BITS} bits)")
    print(f"  Data: {DATA_DIBITS} symbols ({DATA_BITS} bits)")
    print()

    # Find where strong signal starts
    strong_symbols = np.where(np.abs(rx_symbols_decimated) > 0.1)[0]
    if len(strong_symbols) > 0:
        signal_start = strong_symbols[0]
        print(f"Strong signal detected starting at symbol: {signal_start}")
    else:
        signal_start = 0
        print("No strong signal detected, searching from beginning")

    # For short captures (single-shot frame), search from preamble end
    # For long captures, skip ahead to find stable region
    if len(rx_symbols_decimated) < 100000:
        # Short capture - likely single-shot, search after preamble
        search_start = PREAMBLE_DIBITS
        print(f"\nShort capture detected - searching from preamble end (symbol {search_start:,})...")
    elif len(rx_symbols_decimated) < 700000 + 10*FRAME_LEN:
        # Medium capture - search from middle
        search_start = max(signal_start, len(rx_symbols_decimated) // 3)
        print(f"\nSearching from symbol {search_start:,}...")
    else:
        # Long capture - skip far ahead for stable sync
        search_start = 700000
        print(f"\nSkipping to symbol {search_start:,} to find stable synchronized region...")

    print()

    # Get pilot reference from TX (as symbols)
    pilot_bits = tx_bits[PREAMBLE_BITS:PREAMBLE_BITS+PILOT_BITS]
    pilot_ref = bits_to_qpsk_symbols(pilot_bits)

    # Search for pilot with REPEATING structure validation
    print("="*70)
    print("SEARCHING FOR PILOT WITH REPEATING STRUCTURE")
    print("="*70)

    # Leave room for validation, but for short captures use more of the file
    if len(rx_symbols_decimated) < 100000:
        search_end = len(rx_symbols_decimated) - FRAME_LEN  # Just need room for 1 frame
    else:
        search_end = len(rx_symbols_decimated) - 5*FRAME_LEN  # Leave room for 5 frames

    print(f"Searching from symbol {search_start:,} to {search_end:,}...")
    print(f"Pilot: {PILOT_DIBITS} symbols, expecting repeats every {FRAME_LEN} symbols")
    print()

    best_candidates = []

    # Coarse search for pilot
    for pos in range(search_start, search_end, 50):
        if pos % 50000 == 0:
            print(f"  Scanning position {pos:,}...", end='\r')

        rx_pilot = rx_symbols_decimated[pos:pos+PILOT_DIBITS]

        # Check signal power
        power = np.mean(np.abs(rx_pilot))
        if power < 0.3:
            continue

        # Correlate with pilot reference - TEST ALL 4 QPSK ROTATIONS
        rotations = [1, 1j, -1, -1j]
        best_corr = 0
        for rot in rotations:
            test = rx_pilot * rot
            corr_test = np.abs(np.dot(test, np.conj(pilot_ref)))
            if corr_test > best_corr:
                best_corr = corr_test
        corr = best_corr

        # Normalize correlation by pilot energy for fair comparison
        pilot_energy = np.sqrt(np.sum(np.abs(rx_pilot)**2))
        norm_corr = corr / (pilot_energy + 1e-10)

        # Check if this looks like a pilot
        # With rotation-aware correlation, we should get much higher values
        if corr > 40:  # Higher threshold since we test all rotations
            # Verify repeating structure: check next 3 pilots
            repeats_found = 0
            for i in range(1, 4):
                next_pilot_pos = pos + (i * FRAME_LEN)
                if next_pilot_pos + PILOT_DIBITS > len(rx_symbols_decimated):
                    break

                rx_next_pilot = rx_symbols_decimated[next_pilot_pos:next_pilot_pos+PILOT_DIBITS]

                # Test all 4 QPSK rotations for repeat check too
                best_next_corr = 0
                for rot in rotations:
                    test = rx_next_pilot * rot
                    corr_test = np.abs(np.dot(test, np.conj(pilot_ref)))
                    if corr_test > best_next_corr:
                        best_next_corr = corr_test
                next_corr = best_next_corr

                if next_corr > 40:  # Higher threshold with rotation-aware search
                    repeats_found += 1

            # Only accept if we found repeating pilots
            if repeats_found >= 2:
                best_candidates.append({
                    'pos': pos,
                    'corr': corr,
                    'repeats': repeats_found
                })

    print()

    if len(best_candidates) == 0:
        print("✗ NO REPEATING PILOT STRUCTURE FOUND!")
        print()
        print("Possible causes:")
        print("- RX file not captured yet or empty")
        print("- Synchronization completely failed")
        print("- Capture too short - need longer RX time")
        return None

    # Sort by correlation strength
    best_candidates.sort(key=lambda x: x['corr'], reverse=True)
    best = best_candidates[0]

    print(f"✓ Pilot found with repeating structure at position: {best['pos']:,}")
    print(f"  Correlation: {best['corr']:.1f}")
    print(f"  Repeating pilots validated: {best['repeats']}")
    print()

    best_pilot_pos = best['pos']

    # Estimate channel phase offset using the pilot
    rx_pilot = rx_symbols_decimated[best_pilot_pos:best_pilot_pos+PILOT_DIBITS]
    tx_pilot_bits = tx_bits[PREAMBLE_BITS:PREAMBLE_BITS+PILOT_BITS]

    # QPSK has 4-fold ambiguity: test rotations [1, 1j, -1, -1j] = [0°, 90°, 180°, 270°]
    print("Estimating phase offset (checking all QPSK rotations)...")

    best_ber = 1.0
    best_phase_offset = 0
    best_rotation = 1
    best_rotation_name = "0°"

    rotations_test = [(1, "0°"), (1j, "90°"), (-1, "180°"), (-1j, "270°")]

    for rot, rot_name in rotations_test:
        # Apply rotation
        rotated_pilot = rx_pilot * rot

        # Fine phase search ±45°
        for fine_deg in range(-45, 46, 1):
            fine_rad = fine_deg * np.pi / 180

            # Apply fine phase
            test_pilot = rotated_pilot * np.exp(-1j * fine_rad)

            # Demodulate
            test_bits = np.zeros(PILOT_BITS, dtype=np.uint8)
            for i, sym in enumerate(test_pilot):
                test_bits[i*2:i*2+2] = qpsk_symbol_to_bits(sym, 0)

            # Calculate BER
            errors = np.sum(test_bits != tx_pilot_bits)
            ber = errors / PILOT_BITS

            if ber < best_ber:
                best_ber = ber
                best_phase_offset = fine_rad
                best_rotation = rot
                best_rotation_name = rot_name

    print(f"  Best rotation: {best_rotation_name}")
    print(f"  Best phase offset: {best_phase_offset:.4f} rad ({best_phase_offset*180/np.pi:.1f}°)")
    print(f"  Pilot BER: {best_ber:.4f} ({best_ber*100:.1f}%)")
    print()

    # Validate all payloads
    print("="*70)
    print("MULTI-PAYLOAD VALIDATION")
    print("="*70)
    print(f"Frame structure: [Preamble: {PREAMBLE_DIBITS}] + [Pilot: {PILOT_DIBITS} + Data: {DATA_DIBITS}] × {NUM_PAYLOADS}")
    print()

    total_errors = 0
    total_bits = 0
    payloads_good = 0

    # Define QPSK rotations for pilot correlation
    rotations = [1, 1j, -1, -1j]

    for payload_idx in range(NUM_PAYLOADS):
        # Calculate expected position (with drift tolerance)
        expected_pilot_pos = best_pilot_pos + (payload_idx * FRAME_LEN)

        # Search ±30 symbols for pilot (handles drift)
        best_offset = 0
        best_corr = 0

        for offset in range(-30, 31):
            test_pos = expected_pilot_pos + offset

            if test_pos + FRAME_LEN > len(rx_symbols_decimated):
                continue

            rx_pilot = rx_symbols_decimated[test_pos:test_pos+PILOT_DIBITS]

            # Test all 4 QPSK rotations
            test_best_corr = 0
            for rot in rotations:
                test_rot = rx_pilot * rot
                corr_test = np.abs(np.dot(test_rot, np.conj(pilot_ref)))
                if corr_test > test_best_corr:
                    test_best_corr = corr_test
            corr = test_best_corr

            if corr > best_corr:
                best_corr = corr
                best_offset = offset

        # Use best match
        actual_pilot_pos = expected_pilot_pos + best_offset
        data_pos = actual_pilot_pos + PILOT_DIBITS

        if data_pos + DATA_DIBITS > len(rx_symbols_decimated):
            print(f"✗ Payload {payload_idx}: Out of bounds")
            break

        # Get RX data symbols
        rx_data_symbols = rx_symbols_decimated[data_pos:data_pos+DATA_DIBITS]

        # Get TX reference
        tx_data_start = PREAMBLE_BITS + PILOT_BITS + (payload_idx * (PILOT_BITS + DATA_BITS))
        tx_data = tx_bits[tx_data_start:tx_data_start+DATA_BITS]

        # Find best rotation/phase for THIS payload (Costas may have jumped)
        best_payload_ber = 1.0
        best_payload_rotation = 1
        best_payload_phase = 0

        for rot in rotations:
            for phase_deg in range(-45, 46, 5):  # Coarse search for speed
                phase_rad = phase_deg * np.pi / 180

                # Demodulate with this rotation and phase
                test_bits = np.zeros(DATA_BITS, dtype=np.uint8)
                for i, sym in enumerate(rx_data_symbols):
                    corrected = (sym * rot) * np.exp(-1j * phase_rad)
                    test_bits[i*2:i*2+2] = qpsk_symbol_to_bits(corrected, 0)

                errors = np.sum(tx_data != test_bits)
                ber_test = errors / DATA_BITS

                if ber_test < best_payload_ber:
                    best_payload_ber = ber_test
                    best_payload_rotation = rot
                    best_payload_phase = phase_rad

        data_errors = int(best_payload_ber * DATA_BITS)
        ber = best_payload_ber

        total_errors += data_errors
        total_bits += DATA_BITS

        if ber < 0.10:
            payloads_good += 1

        # Status
        status = "✓" if ber < 0.10 else "✗"

        print(f"{status} Payload {payload_idx:2d}: "
              f"errors={data_errors:3d}/{DATA_BITS}, BER={ber:.4f}, "
              f"drift={best_offset:+3d}")

    print()
    print("="*70)
    print("OVERALL RESULTS")
    print("="*70)

    if total_bits == 0:
        print("✗ NO DATA VALIDATED")
        return None

    overall_ber = total_errors / total_bits

    print(f"Total payloads validated: {payloads_good}/{NUM_PAYLOADS}")
    print(f"Total bits: {total_bits}")
    print(f"Total errors: {total_errors}")
    print(f"Overall BER: {overall_ber:.6f} ({overall_ber*100:.2f}%)")
    print()

    if overall_ber == 0:
        print("✓✓✓ PERFECT! No errors!")
    elif overall_ber < 0.01:
        print("✓✓✓ EXCELLENT! BER < 1%")
    elif overall_ber < 0.05:
        print("✓✓ VERY GOOD! BER < 5%")
    elif overall_ber < 0.10:
        print("✓ GOOD! BER < 10%")
    elif overall_ber < 0.20:
        print("⚠ MARGINAL - BER 10-20%")
    else:
        print("✗ POOR - BER > 20%")

    print("="*70)
    print()

    return {
        'ber': overall_ber,
        'total_errors': total_errors,
        'total_bits': total_bits,
        'payloads_good': payloads_good,
        'num_payloads': NUM_PAYLOADS
    }
