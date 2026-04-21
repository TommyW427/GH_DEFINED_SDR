#!/usr/bin/env python3
"""
BER/SER Validation
==================

Advanced validation for received frames with multi-payload support.
"""

import numpy as np
import json
from pathlib import Path


def validate_frame(tx_frame_file, rx_symbols_file, sps=1, metadata_file=None):
    """
    Validate BER and SER for received frame with full payload analysis

    Args:
        tx_frame_file: Transmitted frame bits (e.g., 'full_frames/frame_0000.bin')
        rx_symbols_file: Received symbols from SPV.py ('received_symbols.bin')
        sps: Additional decimation (default 1, since symbol_sync already outputs at symbol rate)
        metadata_file: Optional JSON metadata file (default: 'frames/test_frame_metadata.json')

    Returns:
        dict with BER, SER, per-payload analysis, and detailed results
    """
    # Load metadata
    if metadata_file is None:
        metadata_file = 'frames/test_frame_metadata.json'

    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        PREAMBLE_LEN = metadata['preamble_bits']
        PILOT_LEN = metadata['pilot_bits']
        DATA_LEN = metadata['data_bits']
        NUM_PAYLOADS = metadata['num_payloads']
        PAYLOAD_FILES = metadata['payload_files']
    except Exception as e:
        print(f"Warning: Could not load metadata from {metadata_file}: {e}")
        print("Using default values...")
        PREAMBLE_LEN = 2000
        PILOT_LEN = 64
        DATA_LEN = 64
        NUM_PAYLOADS = 10
        PAYLOAD_FILES = [f"payloads/data_{i:03d}.bin" for i in range(NUM_PAYLOADS)]

    # Load transmitted bits
    tx_bits = np.fromfile(tx_frame_file, dtype=np.uint8)

    # Load received symbols (already at symbol rate from GNU Radio symbol_sync)
    rx_symbols = np.fromfile(rx_symbols_file, dtype=np.complex64)

    # No decimation needed - symbol_sync already did it
    # But keep variable for compatibility
    rx_symbols_decimated = rx_symbols if sps == 1 else rx_symbols[::sps]

    # Remove DC offset by centering around mean
    mean_real = np.mean(rx_symbols_decimated.real)
    rx_symbols_centered = rx_symbols_decimated.real - mean_real

    # Hard decision: convert symbols to bits (BPSK: centered real > 0 => 1, else 0)
    rx_bits = (rx_symbols_centered > 0).astype(np.uint8)

    # Frame structure
    FRAME_LEN = PILOT_LEN + DATA_LEN  # Pilot + Data = one complete frame unit

    print("="*70)
    print("FRAME VALIDATION")
    print("="*70)
    print(f"TX bits: {len(tx_bits)}")
    print(f"RX symbols (raw): {len(rx_symbols)}")
    print(f"RX symbols (decimated): {len(rx_symbols_decimated)}")
    print(f"RX bits: {len(rx_bits)}")
    print(f"SPS: {sps}")
    print()

    # Find where strong signal starts (skip silence at beginning)
    strong_symbols = np.where(np.abs(rx_symbols_centered) > 0.1)[0]
    if len(strong_symbols) > 0:
        signal_start = strong_symbols[0]
        print(f"Strong signal detected starting at symbol: {signal_start}")
    else:
        signal_start = 0
        print("No strong signal detected, searching from beginning")

    # IMPORTANT: Find the BEST synchronized region, not just the first signal!
    # The synchronization loops need time to converge, so the best data is LATER
    # We'll scan through the capture and find regions with good BPSK characteristics:
    #   1. Balanced positive/negative symbols (40-60% each)
    #   2. High variance (two distinct clusters, not stuck at one value)
    #   3. Strong signal power

    print("\nScanning for best synchronized region (where loops have converged)...")

    # IMPORTANT: Skip far ahead to avoid convergence period!
    # The first ~700k symbols may be during loop convergence
    # We want to analyze data where everything is FULLY locked
    min_start_position = 700000
    actual_start = max(signal_start, min_start_position)

    print(f"Skipping to symbol {actual_start:,} to ensure full convergence...")

    best_regions = []
    window_size = 2000  # Scan in 2000-symbol windows
    step_size = 500     # Move window by 500 symbols each time

    for start_pos in range(actual_start, len(rx_symbols_centered) - window_size, step_size):
        window = rx_symbols_centered[start_pos:start_pos+window_size]

        # Calculate quality metrics
        power = np.mean(np.abs(rx_symbols_decimated[start_pos:start_pos+window_size]))
        variance = np.var(window)
        positive_ratio = np.sum(window > 0) / len(window)

        # Good BPSK should have:
        # - Balanced symbols (positive ratio between 0.3 and 0.7)
        # - High variance (distinct clusters, not stuck)
        # - Decent power (>0.05)

        quality_score = 0
        if 0.3 < positive_ratio < 0.7:  # Balanced
            quality_score += 100 * (0.5 - abs(0.5 - positive_ratio))  # Higher score = closer to 50/50
        quality_score += variance * 1000  # High variance is good
        quality_score += power * 10       # Decent power is good

        if power > 0.05:  # Only consider regions with decent signal
            best_regions.append({
                'start': start_pos,
                'end': start_pos + window_size,
                'power': power,
                'variance': variance,
                'positive_ratio': positive_ratio,
                'quality_score': quality_score
            })

    # Sort by quality score and show top candidates
    best_regions.sort(key=lambda x: x['quality_score'], reverse=True)

    print(f"\nTop 5 best synchronized regions (out of {len(best_regions)} scanned):")
    for i, region in enumerate(best_regions[:5]):
        print(f"  {i+1}. Position {region['start']:6d}-{region['end']:6d}: "
              f"power={region['power']:.3f}, var={region['variance']:.4f}, "
              f"pos%={region['positive_ratio']*100:.1f}%, quality={region['quality_score']:.1f}")

    # Use the best region for analysis
    if best_regions:
        best_region = best_regions[0]
        analysis_start = best_region['start']
        signal_region = rx_symbols_centered[analysis_start:analysis_start+1000]

        print(f"\nUsing best synchronized region starting at symbol: {analysis_start}")
        print("\nSymbol statistics in BEST region:")
        print(f"  Mean: {np.mean(signal_region):.4f}")
        print(f"  Std Dev: {np.std(signal_region):.4f}")
        print(f"  Min: {np.min(signal_region):.4f}")
        print(f"  Max: {np.max(signal_region):.4f}")
        print(f"  Variance: {np.var(signal_region):.4f}")

        # For BPSK, we expect two clusters around -A and +A
        positive_count = np.sum(signal_region > 0)
        negative_count = np.sum(signal_region < 0)
        print(f"  Positive symbols: {positive_count} ({100*positive_count/len(signal_region):.1f}%)")
        print(f"  Negative symbols: {negative_count} ({100*negative_count/len(signal_region):.1f}%)")

        # Update signal_start to use the best region
        signal_start = analysis_start
    else:
        print("\nWarning: No good synchronized regions found!")
        print("Using original signal start position.")

    print()

    # Check for repeating patterns in received data (autocorrelation)
    # This tests if the RX symbols themselves repeat every FRAME_LEN, regardless of TX
    print("Checking for repeating patterns in received data...")

    # Take a window of received bits starting from signal start
    window_start = signal_start
    window_size = min(2000, len(rx_bits) - window_start - FRAME_LEN*5)  # Check 5 frames worth

    if window_size > FRAME_LEN * 3:
        rx_window = rx_bits[window_start:window_start+window_size]

        # Check autocorrelation at FRAME_LEN spacing (128 symbols)
        autocorr_scores = []
        for lag in range(FRAME_LEN-10, FRAME_LEN+10):  # Check around expected spacing
            if window_start + lag + window_size <= len(rx_bits):
                rx_shifted = rx_bits[window_start+lag:window_start+lag+window_size]
                # Calculate agreement between original and shifted
                agreement = np.sum(rx_window == rx_shifted) / len(rx_window)
                autocorr_scores.append({'lag': lag, 'agreement': agreement})

        # Find best autocorrelation
        autocorr_scores.sort(key=lambda x: x['agreement'], reverse=True)
        best_lag = autocorr_scores[0]

        print(f"Best autocorrelation at lag={best_lag['lag']}, agreement={best_lag['agreement']:.4f}")
        print(f"Expected lag: {FRAME_LEN}")
        print(f"Top 5 autocorrelation lags:")
        for i, lag_info in enumerate(autocorr_scores[:5]):
            print(f"  {i+1}. lag={lag_info['lag']:3d}, agreement={lag_info['agreement']:.4f}")

        if best_lag['agreement'] > 0.8:
            print(f"✓ Strong repeating pattern detected! Symbols repeat every {best_lag['lag']} positions")
        elif best_lag['agreement'] > 0.6:
            print(f"⚠ Moderate repeating pattern at {best_lag['lag']} positions")
        else:
            print(f"✗ No clear repeating pattern (possible timing drift or noise)")
        print()
    else:
        print("Not enough data to check autocorrelation")
        print()

    # ========================================================================
    # FIND PREAMBLE FIRST (not pilot!)
    # ========================================================================
    # The key insight: we need to find the FRAME, not just any pilot
    # A frame is: [Preamble: 6000] + [Pilot+Data] × 10
    # We should correlate with the preamble to find frame boundaries!

    print("="*70)
    print("SEARCHING FOR FRAME USING PREAMBLE CORRELATION")
    print("="*70)

    preamble_ref = tx_bits[:PREAMBLE_LEN]
    preamble_ref_bipolar = 2 * preamble_ref.astype(np.float32) - 1

    # Search for preamble in the BEST region we identified
    preamble_search_start = signal_start
    preamble_search_end = min(len(rx_bits) - PREAMBLE_LEN - 10*FRAME_LEN, signal_start + 100000)

    print(f"Searching for preamble from {preamble_search_start:,} to {preamble_search_end:,}")
    print(f"Preamble length: {PREAMBLE_LEN} bits")
    print()

    best_preamble_corr = -1
    best_preamble_pos = None
    best_preamble_inverted = False

    # Use FULL correlation (not subsampled) for more accurate detection
    print("Computing correlation with preamble (this may take a moment)...")

    # Prepare RX data for correlation
    rx_search_region = rx_bits[preamble_search_start:preamble_search_end]

    if len(rx_search_region) > PREAMBLE_LEN:
        # Compute sliding correlation
        correlations = []
        for pos in range(0, len(rx_search_region) - PREAMBLE_LEN, 1):
            rx_window = rx_search_region[pos:pos+PREAMBLE_LEN]
            rx_window_bipolar = 2 * rx_window.astype(np.float32) - 1

            # Correlation (try both normal and inverted)
            corr_normal = np.sum(preamble_ref_bipolar * rx_window_bipolar) / PREAMBLE_LEN
            corr_inv = np.sum(preamble_ref_bipolar * (-rx_window_bipolar)) / PREAMBLE_LEN

            # Take absolute value and store best
            if abs(corr_normal) > abs(corr_inv):
                correlations.append((pos + preamble_search_start, abs(corr_normal), False))
            else:
                correlations.append((pos + preamble_search_start, abs(corr_inv), True))

        # Find best correlation
        correlations.sort(key=lambda x: x[1], reverse=True)

        if correlations:
            best_preamble_pos, best_preamble_corr, best_preamble_inverted = correlations[0]

            # Show top 5 preamble candidates
            print("\nTop 5 preamble candidates:")
            for i, (pos, corr, inv) in enumerate(correlations[:5]):
                print(f"  {i+1}. pos={pos:,}, corr={corr:.4f}, inverted={inv}")
            print()

    print(f"Best preamble found at position: {best_preamble_pos:,}")
    print(f"Preamble correlation: {best_preamble_corr:.4f}")
    print(f"Phase inverted: {best_preamble_inverted}")
    print()

    if best_preamble_corr < 0.5:
        print("WARNING: Preamble correlation is low! Frame detection may be unreliable.")
        print(f"Expected >0.8, got {best_preamble_corr:.4f}")
        print()

        # DIAGNOSTIC: Let's figure out WHY correlation is low
        print("="*70)
        print("DIAGNOSTIC: Analyzing preamble mismatch")
        print("="*70)

        # Extract received preamble at best position
        rx_preamble = rx_bits[best_preamble_pos:best_preamble_pos+PREAMBLE_LEN]
        if best_preamble_inverted:
            rx_preamble = 1 - rx_preamble

        # Compare with reference
        preamble_errors = np.sum(preamble_ref != rx_preamble)
        preamble_agreement = 1.0 - (preamble_errors / PREAMBLE_LEN)

        print(f"Bit errors in preamble: {preamble_errors}/{PREAMBLE_LEN}")
        print(f"Bit agreement: {preamble_agreement:.4f} ({preamble_agreement*100:.2f}%)")
        print()

        # Show first 100 bits comparison
        print("First 100 bits comparison:")
        print(f"  TX: {preamble_ref[:100]}")
        print(f"  RX: {rx_preamble[:100]}")
        print()

        # Check if there's a systematic issue
        # 1. Check if RX preamble is all zeros or all ones
        rx_ones = np.sum(rx_preamble)
        rx_zeros = PREAMBLE_LEN - rx_ones
        print(f"RX preamble statistics:")
        print(f"  Ones:  {rx_ones} ({rx_ones/PREAMBLE_LEN*100:.1f}%)")
        print(f"  Zeros: {rx_zeros} ({rx_zeros/PREAMBLE_LEN*100:.1f}%)")
        print()

        # 2. Check TX preamble for comparison
        tx_ones = np.sum(preamble_ref)
        tx_zeros = PREAMBLE_LEN - tx_ones
        print(f"TX preamble statistics:")
        print(f"  Ones:  {tx_ones} ({tx_ones/PREAMBLE_LEN*100:.1f}%)")
        print(f"  Zeros: {tx_zeros} ({tx_zeros/PREAMBLE_LEN*100:.1f}%)")
        print()

        # 3. Check if there's a delay/offset issue
        print("Checking for timing offset in preamble match...")
        best_offset_corr = best_preamble_corr
        best_offset = 0

        for offset in range(-50, 51):
            if best_preamble_pos + offset < 0 or best_preamble_pos + offset + PREAMBLE_LEN > len(rx_bits):
                continue

            rx_preamble_offset = rx_bits[best_preamble_pos + offset:best_preamble_pos + offset + PREAMBLE_LEN]
            if best_preamble_inverted:
                rx_preamble_offset = 1 - rx_preamble_offset

            errors = np.sum(preamble_ref != rx_preamble_offset)
            agreement = 1.0 - (errors / PREAMBLE_LEN)

            if agreement > best_offset_corr:
                best_offset_corr = agreement
                best_offset = offset

        if best_offset != 0:
            print(f"  Better match found at offset {best_offset:+d} symbols")
            print(f"  Agreement: {best_offset_corr:.4f} (vs {best_preamble_corr:.4f} at offset 0)")
            print(f"  Should update preamble position to: {best_preamble_pos + best_offset:,}")

            # Update position
            best_preamble_pos += best_offset
            best_preamble_corr = best_offset_corr
        else:
            print(f"  No better match found within ±50 symbols")

        print()
        print("="*70)
        print()

    # Now search for pilots WITHIN this frame (starting right after preamble)
    frame_start = best_preamble_pos
    first_pilot_pos = frame_start + PREAMBLE_LEN

    print(f"Frame starts at: {frame_start:,}")
    print(f"First pilot should be at: {first_pilot_pos:,}")
    print(f"Frame should end at: {frame_start + PREAMBLE_LEN + NUM_PAYLOADS * FRAME_LEN:,}")
    print()

    # Find pilot using correlation
    pilot_ref = tx_bits[PREAMBLE_LEN:PREAMBLE_LEN+PILOT_LEN]

    # Convert to +1/-1 for correlation (better than 0/1)
    pilot_ref_bipolar = 2 * pilot_ref.astype(np.float32) - 1

    # Search for first pilot in a MUCH WIDER region
    # Since preamble detection is unreliable, we need to search broadly
    candidates = []

    # Search in a very wide window around the best synchronized region
    # Use the signal_start from the quality scan, not the unreliable preamble position
    search_start = signal_start - 10000  # Search 10k symbols before best region
    search_end = signal_start + 50000     # Search 50k symbols after

    # Ensure we don't go out of bounds
    search_start = max(0, search_start)
    search_end = min(len(rx_bits) - PILOT_LEN, search_end)

    print(f"Searching for pilot in WIDE region from position {search_start:,} to {search_end:,}")
    print(f"Search region size: {search_end - search_start:,} positions")
    print(f"Pilot reference (first 20 bits): {pilot_ref[:20]}")
    print()

    # Calculate signal strength threshold - pilots should be strong
    # Use the symbols themselves (before hard decision) for power measurement
    pilot_power_threshold = 0.01  # Minimum average power for valid pilot region

    print(f"Pilot power threshold: {pilot_power_threshold:.4f}")
    print(f"Searching with signal strength requirement...\n")

    # Search every position (step=1 for accurate alignment)
    for pos in range(search_start, search_end):
        rx_window = rx_bits[pos:pos+PILOT_LEN]
        if len(rx_window) < PILOT_LEN:
            break

        # Check signal strength at this position using the original symbols
        rx_symbols_window = rx_symbols_decimated[pos:pos+PILOT_LEN]
        avg_power = np.mean(np.abs(rx_symbols_window))

        # Skip this position if signal is too weak (likely noise or off-signal region)
        if avg_power < pilot_power_threshold:
            continue

        # Convert RX bits to +1/-1 for correlation
        rx_window_bipolar = 2 * rx_window.astype(np.float32) - 1

        # Normal correlation
        corr = np.sum(pilot_ref_bipolar * rx_window_bipolar) / PILOT_LEN
        errs = np.sum(pilot_ref != rx_window)
        candidates.append({
            'pos': pos,
            'errors': errs,
            'inverted': False,
            'correlation': corr,
            'power': avg_power
        })

        # Inverted correlation (180° phase)
        corr_inv = np.sum(pilot_ref_bipolar * (-rx_window_bipolar)) / PILOT_LEN
        errs_inv = np.sum(pilot_ref != (1 - rx_window))
        candidates.append({
            'pos': pos,
            'errors': errs_inv,
            'inverted': True,
            'correlation': corr_inv,
            'power': avg_power
        })

    # Sort by errors (fewest first)
    candidates.sort(key=lambda x: x['errors'])

    print(f"Found {len(candidates)} candidate positions with sufficient signal strength")
    print()

    # Show top 10 candidates to see if we have repeating pattern
    print("Top 10 pilot candidates (with signal power):")
    for i, cand in enumerate(candidates[:10]):
        print(f"  {i+1}. pos={cand['pos']:6d}, errors={cand['errors']:2d}/{PILOT_LEN}, "
              f"corr={cand['correlation']:+.4f}, power={cand['power']:.4f}, inverted={cand['inverted']}")
    print()

    # Check for repeating pilots (should be FRAME_LEN apart)
    # Find all candidates with low errors (< 10 errors)
    good_candidates = [c for c in candidates if c['errors'] < 10]

    if len(good_candidates) >= 2:
        print(f"Found {len(good_candidates)} candidates with <10 errors")

        # Check spacing between good candidates
        positions = sorted([c['pos'] for c in good_candidates])
        if len(positions) >= 2:
            spacing = np.diff(positions)
            print(f"Spacing between good candidates: {spacing[:10]}")
            # Expected spacing should be FRAME_LEN (128)
            if len(spacing) > 0:
                median_spacing = int(np.median(spacing))
                print(f"Median spacing: {median_spacing} (expected: {FRAME_LEN})")
                if abs(median_spacing - FRAME_LEN) < 5:
                    print("✓ Spacing matches frame structure!")
                else:
                    print(f"⚠ Spacing mismatch! Off by {median_spacing - FRAME_LEN} symbols")
        print()

    # Use best candidate as the FIRST pilot
    best = candidates[0]

    print(f"Best pilot found at position: {best['pos']:,}")
    print(f"Pilot errors: {best['errors']}/{PILOT_LEN}")
    print(f"Pilot correlation: {best['correlation']:.4f}")
    print(f"Phase inverted: {best['inverted']}")
    print(f"Signal power: {best['power']:.3f}")
    print()

    # Now verify this is actually the FIRST pilot by checking frame structure
    # Expected: this pilot should be followed by DATA, then PILOT, then DATA, etc.
    # at intervals of FRAME_LEN (128 symbols)

    print("Verifying frame structure from this pilot position...")
    first_pilot_candidate = best['pos']

    # Check the next few pilot positions (should be at +128, +256, +384, etc.)
    pilot_spacing_check = []
    for i in range(1, min(5, NUM_PAYLOADS)):  # Check up to 4 more pilots
        next_pilot_pos = first_pilot_candidate + (i * FRAME_LEN)

        if next_pilot_pos + PILOT_LEN <= len(rx_bits):
            next_pilot_rx = rx_bits[next_pilot_pos:next_pilot_pos+PILOT_LEN]
            if best['inverted']:
                next_pilot_rx = 1 - next_pilot_rx

            errors = np.sum(pilot_ref != next_pilot_rx)
            pilot_spacing_check.append({
                'index': i,
                'position': next_pilot_pos,
                'errors': errors,
                'expected_position': first_pilot_candidate + (i * FRAME_LEN)
            })

            print(f"  Pilot {i}: pos={next_pilot_pos:,}, errors={errors}/64")

    # Calculate average pilot errors across the frame
    avg_pilot_errors = np.mean([p['errors'] for p in pilot_spacing_check])
    print(f"\nAverage pilot errors: {avg_pilot_errors:.1f}/64")

    if avg_pilot_errors < 20:
        print("✓ Frame structure looks good! Pilots are consistently matched.")
    else:
        print("⚠ Frame structure uncertain - pilot matches are weak.")
    print()

    # Show received pilot for debugging
    if best['inverted']:
        rx_pilot_corrected = 1 - rx_bits[best['pos']:best['pos']+PILOT_LEN]
    else:
        rx_pilot_corrected = rx_bits[best['pos']:best['pos']+PILOT_LEN]
    print(f"RX pilot (first 20 bits): {rx_pilot_corrected[:20]}")
    print()

    # Apply phase correction
    if best['inverted']:
        rx_bits_corrected = 1 - rx_bits
    else:
        rx_bits_corrected = rx_bits

    # ========================================================================
    # ENHANCED: Validate ALL payloads in the frame
    # ========================================================================

    print("="*70)
    print("MULTI-PAYLOAD VALIDATION")
    print("="*70)
    print(f"Frame structure: [Preamble: {PREAMBLE_LEN}] + [Pilot: {PILOT_LEN} + Data: {DATA_LEN}] × {NUM_PAYLOADS}")
    print()

    # Load all payload reference files
    payload_references = []
    for i, payload_file in enumerate(PAYLOAD_FILES):
        try:
            payload_bits = np.fromfile(payload_file, dtype=np.uint8)
            if len(payload_bits) != DATA_LEN:
                print(f"Warning: Payload {i} has {len(payload_bits)} bits, expected {DATA_LEN}")
            payload_references.append(payload_bits)
        except Exception as e:
            print(f"Warning: Could not load {payload_file}: {e}")
            payload_references.append(None)

    # First pilot position is at best['pos']
    # Each subsequent [Pilot+Data] pair is FRAME_LEN apart
    first_pilot_pos = best['pos']

    payload_results = []
    total_payload_bits = 0
    total_payload_errors = 0

    for payload_idx in range(NUM_PAYLOADS):
        # Calculate position of this payload's pilot and data
        pilot_pos = first_pilot_pos + (payload_idx * FRAME_LEN)
        data_pos = pilot_pos + PILOT_LEN

        # Check if we have enough received data
        if data_pos + DATA_LEN > len(rx_bits_corrected):
            print(f"Payload {payload_idx}: Insufficient RX data (needed {data_pos + DATA_LEN}, have {len(rx_bits_corrected)})")
            payload_results.append({
                'payload_index': payload_idx,
                'status': 'insufficient_data',
                'ber': None,
                'errors': None
            })
            continue

        # Extract received pilot and data for this payload
        rx_pilot = rx_bits_corrected[pilot_pos:pilot_pos+PILOT_LEN]
        rx_data = rx_bits_corrected[data_pos:data_pos+DATA_LEN]

        # Get transmitted pilot and data
        # Pilot is always the same (from preamble section in tx_bits)
        tx_pilot = tx_bits[PREAMBLE_LEN:PREAMBLE_LEN+PILOT_LEN]

        # Data depends on which payload we're looking at
        # In the transmitted frame: [Preamble][Pilot+Data0][Pilot+Data1]...[Pilot+Data9]
        tx_data_start = PREAMBLE_LEN + PILOT_LEN + (payload_idx * FRAME_LEN)
        tx_data = tx_bits[tx_data_start:tx_data_start+DATA_LEN]

        # Validate pilot alignment
        pilot_errors = np.sum(tx_pilot != rx_pilot)
        pilot_ber = pilot_errors / PILOT_LEN

        # Validate data
        data_errors = np.sum(tx_data != rx_data)
        data_ber = data_errors / DATA_LEN

        # Try to identify which payload this is (cross-check with reference files)
        if payload_references[payload_idx] is not None:
            ref_data = payload_references[payload_idx]
            ref_errors = np.sum(ref_data != rx_data)
            ref_ber = ref_errors / DATA_LEN
            matches_reference = (ref_errors == data_errors)
        else:
            ref_errors = None
            ref_ber = None
            matches_reference = None

        # Store results
        payload_results.append({
            'payload_index': payload_idx,
            'status': 'ok',
            'pilot_position': pilot_pos,
            'data_position': data_pos,
            'pilot_errors': pilot_errors,
            'pilot_ber': pilot_ber,
            'data_errors': data_errors,
            'data_ber': data_ber,
            'matches_reference': matches_reference,
            'tx_data_sample': tx_data[:10].tolist(),
            'rx_data_sample': rx_data[:10].tolist()
        })

        total_payload_bits += DATA_LEN
        total_payload_errors += data_errors

        # Print per-payload results
        status_symbol = "✓" if data_ber < 0.01 else ("⚠" if data_ber < 0.10 else "✗")
        match_str = "" if matches_reference is None else (
            " [REF ✓]" if matches_reference else " [REF ✗]"
        )

        print(f"{status_symbol} Payload {payload_idx:2d}: "
              f"pilot@{pilot_pos:6d} (errs={pilot_errors:2d}/{PILOT_LEN}, BER={pilot_ber:.4f}), "
              f"data@{data_pos:6d} (errs={data_errors:2d}/{DATA_LEN}, BER={data_ber:.4f}){match_str}")

    print()

    # Overall statistics
    overall_ber = total_payload_errors / total_payload_bits if total_payload_bits > 0 else None

    # Count perfect/good/poor payloads
    perfect_count = sum(1 for p in payload_results if p.get('data_ber') == 0.0)
    good_count = sum(1 for p in payload_results if p.get('data_ber', 1.0) > 0.0 and p.get('data_ber', 1.0) < 0.01)
    marginal_count = sum(1 for p in payload_results if p.get('data_ber', 1.0) >= 0.01 and p.get('data_ber', 1.0) < 0.10)
    poor_count = sum(1 for p in payload_results if p.get('data_ber', 1.0) >= 0.10)

    print("="*70)
    print("OVERALL RESULTS")
    print("="*70)
    print(f"Total payloads validated: {len([p for p in payload_results if p['status'] == 'ok'])}/{NUM_PAYLOADS}")
    print(f"Total data bits compared: {total_payload_bits}")
    print(f"Total bit errors: {total_payload_errors}")
    print(f"Overall BER: {overall_ber:.6f} ({overall_ber*100:.4f}%)" if overall_ber is not None else "Overall BER: N/A")
    print()
    print(f"Payload Quality Breakdown:")
    print(f"  Perfect (BER = 0%):       {perfect_count}/{NUM_PAYLOADS}")
    print(f"  Excellent (BER < 1%):     {good_count}/{NUM_PAYLOADS}")
    print(f"  Marginal (1% ≤ BER < 10%): {marginal_count}/{NUM_PAYLOADS}")
    print(f"  Poor (BER ≥ 10%):         {poor_count}/{NUM_PAYLOADS}")
    print()

    # First pilot check (original validation for backward compatibility)
    first_payload = payload_results[0] if len(payload_results) > 0 else None

    if overall_ber is not None:
        if overall_ber == 0:
            print("✓✓✓ PERFECT! Zero errors across all payloads!")
        elif overall_ber < 0.01:
            print("✓✓ EXCELLENT! Overall BER < 1%")
        elif overall_ber < 0.05:
            print("✓ GOOD - Overall BER < 5%")
        elif overall_ber < 0.10:
            print("⚠ MARGINAL - Overall BER 5-10%")
        else:
            print("✗ POOR - Overall BER > 10%")

    print("="*70)

    # Calculate SER (symbol errors)
    # For BPSK, 1 symbol = 1 bit, so SER = BER
    ser = overall_ber

    return {
        'ber': overall_ber,
        'ser': ser,
        'bit_errors': total_payload_errors,
        'total_bits': total_payload_bits,
        'pilot_errors': best['errors'],
        'pilot_position': best['pos'],
        'phase_inverted': best['inverted'],
        'num_payloads': NUM_PAYLOADS,
        'payloads_validated': len([p for p in payload_results if p['status'] == 'ok']),
        'payload_results': payload_results,
        'perfect_payloads': perfect_count,
        'good_payloads': good_count,
        'marginal_payloads': marginal_count,
        'poor_payloads': poor_count
    }
