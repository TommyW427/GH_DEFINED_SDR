#!/usr/bin/env python3
"""
Production validator for oversampled captures using the post-recovery stream.

This started as the QPSK post-Costas validator, but now supports BPSK, QPSK,
16QAM, and 64QAM by reading modulation details from frame metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from core.modulation import bits_per_symbol, bits_to_symbols, symbols_to_bits


def load_metadata(metadata_path: Path) -> dict:
    with metadata_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_expected_payloads(metadata: dict, root: Path, tx_bits: np.ndarray) -> list[np.ndarray]:
    payload_bits_len = metadata["data_bits"]
    payloads = []
    for idx in range(metadata["num_payloads"]):
        payload_files = metadata.get("payload_files", [])
        if idx < len(payload_files):
            payload_path = Path(payload_files[idx])
            if not payload_path.is_absolute():
                payload_path = root / payload_path
            if payload_path.exists():
                payload = np.fromfile(payload_path, dtype=np.uint8)[:payload_bits_len]
                if len(payload) < payload_bits_len:
                    payload = np.pad(payload, (0, payload_bits_len - len(payload)))
                payloads.append(payload.astype(np.uint8))
                continue
        tx_start = (
            metadata["preamble_bits"]
            + metadata["pilot_bits"]
            + idx * (metadata["pilot_bits"] + metadata["data_bits"])
        )
        payloads.append(tx_bits[tx_start:tx_start + payload_bits_len].astype(np.uint8))
    return payloads


def averaged_symbols(samples: np.ndarray, sps: int, phase: int) -> np.ndarray:
    usable = samples[phase:]
    usable = usable[: (len(usable) // sps) * sps]
    return usable.reshape(-1, sps).mean(axis=1)


def oversampled_symbol_reference(symbols: np.ndarray, sps: int) -> np.ndarray:
    if len(symbols) == 0:
        return np.empty(0, dtype=np.complex64)
    return np.repeat(np.asarray(symbols, dtype=np.complex64), sps)


def pilot_score(rx_pilot: np.ndarray, tx_pilot: np.ndarray) -> float:
    denom = np.linalg.norm(rx_pilot) * np.linalg.norm(tx_pilot)
    if denom < 1e-12:
        return 0.0
    return float(np.abs(np.vdot(tx_pilot, rx_pilot)) / denom)


def segment_score(rx_slice: np.ndarray, tx_ref: np.ndarray) -> float:
    return pilot_score(rx_slice, tx_ref)


def differential_segment_score(rx_slice: np.ndarray, tx_ref: np.ndarray) -> float:
    if len(rx_slice) < 2 or len(tx_ref) < 2:
        return segment_score(rx_slice, tx_ref)
    rx_diff = rx_slice[1:] * np.conj(rx_slice[:-1])
    tx_diff = tx_ref[1:] * np.conj(tx_ref[:-1])
    denom = np.linalg.norm(rx_diff) * np.linalg.norm(tx_diff)
    if denom < 1e-12:
        return 0.0
    return float(np.abs(np.vdot(tx_diff, rx_diff)) / denom)


def repeated_pilot_score(
    rx_symbols: np.ndarray,
    pilot_pos: int,
    pilot_ref: np.ndarray,
    frame_symbols: int,
    num_payloads: int,
    pilots_to_check: int,
) -> tuple[float, int]:
    scores = []
    checked = 0
    for idx in range(min(num_payloads, pilots_to_check)):
        pos = pilot_pos + idx * frame_symbols
        sl = rx_symbols[pos:pos + len(pilot_ref)]
        if len(sl) != len(pilot_ref):
            break
        checked += 1
        scores.append(pilot_score(sl, pilot_ref))
    return (float(np.mean(scores)) if scores else 0.0, checked)


def find_frame_start(
    rx_symbols: np.ndarray,
    preamble_symbols: int,
    pilot_ref: np.ndarray,
    frame_symbols: int,
    num_payloads: int,
    search_start: int,
    coarse_step: int,
    refine_radius: int,
    pilots_to_check: int,
) -> dict:
    search_end = len(rx_symbols) - (preamble_symbols + frame_symbols * num_payloads) + 1
    if search_end <= search_start:
        raise ValueError("Capture is too short for one full frame.")

    best = None
    for frame_start in range(search_start, search_end, coarse_step):
        pilot_pos = frame_start + preamble_symbols
        score, checked = repeated_pilot_score(
            rx_symbols, pilot_pos, pilot_ref, frame_symbols, num_payloads, pilots_to_check
        )
        if best is None or score > best["score"]:
            best = {"frame_start": frame_start, "pilot_start": pilot_pos, "score": score, "checked": checked}

    refined = best
    lo = max(search_start, best["frame_start"] - refine_radius)
    hi = min(search_end, best["frame_start"] + refine_radius + 1)
    for frame_start in range(lo, hi):
        pilot_pos = frame_start + preamble_symbols
        score, checked = repeated_pilot_score(
            rx_symbols, pilot_pos, pilot_ref, frame_symbols, num_payloads, pilots_to_check
        )
        if score > refined["score"]:
            refined = {"frame_start": frame_start, "pilot_start": pilot_pos, "score": score, "checked": checked}
    return refined


def find_frame_start_from_preamble(
    rx_symbols: np.ndarray,
    preamble_ref: np.ndarray,
    search_start: int,
    coarse_step: int,
    refine_radius: int,
    differential: bool = False,
) -> dict:
    search_end = len(rx_symbols) - len(preamble_ref) + 1
    if search_end <= search_start:
        raise ValueError("Capture is too short for one full preamble.")

    score_fn = differential_segment_score if differential else segment_score
    best = None
    for frame_start in range(search_start, search_end, coarse_step):
        sl = rx_symbols[frame_start:frame_start + len(preamble_ref)]
        score = score_fn(sl, preamble_ref)
        if best is None or score > best["score"]:
            best = {"frame_start": frame_start, "score": score}

    refined = best
    lo = max(search_start, best["frame_start"] - refine_radius)
    hi = min(search_end, best["frame_start"] + refine_radius + 1)
    for frame_start in range(lo, hi):
        sl = rx_symbols[frame_start:frame_start + len(preamble_ref)]
        score = score_fn(sl, preamble_ref)
        if score > refined["score"]:
            refined = {"frame_start": frame_start, "score": score}
    return refined


def estimate_sync_rotation(rx_sync: np.ndarray, tx_sync: np.ndarray) -> tuple[float, float]:
    if len(rx_sync) == 0 or len(tx_sync) == 0:
        return 0.0, 0.0

    matched = rx_sync * np.conj(tx_sync)
    phase0 = float(np.angle(np.mean(matched)))

    if len(rx_sync) < 2 or len(tx_sync) < 2:
        return 0.0, phase0

    rx_diff = rx_sync[1:] * np.conj(rx_sync[:-1])
    tx_diff = tx_sync[1:] * np.conj(tx_sync[:-1])
    diff_match = rx_diff * np.conj(tx_diff)
    omega = float(np.angle(np.mean(diff_match)))
    return omega, phase0


def derotate_symbols(rx_symbols: np.ndarray, omega: float, phase0: float) -> np.ndarray:
    if len(rx_symbols) == 0:
        return rx_symbols
    n = np.arange(len(rx_symbols), dtype=np.float64)
    correction = np.exp(-1j * (phase0 + omega * n))
    return (rx_symbols * correction).astype(np.complex64)


def estimate_sync_rotation_from_oversampled(
    samples: np.ndarray,
    sync_ref: np.ndarray,
    sps: int,
    phase: int,
    sync_start_symbols: int,
) -> tuple[float, float]:
    sample_start = int(phase + sync_start_symbols * sps)
    sync_ref_os = oversampled_symbol_reference(sync_ref, sps)
    if sample_start < 0 or sample_start + len(sync_ref_os) > len(samples):
        return 0.0, 0.0
    rx_sync_os = samples[sample_start:sample_start + len(sync_ref_os)]
    if len(rx_sync_os) != len(sync_ref_os):
        return 0.0, 0.0
    return estimate_sync_rotation(rx_sync_os, sync_ref_os)


def nearest_pilot(rx_symbols: np.ndarray, pilot_ref: np.ndarray, expected_pos: int, radius: int) -> tuple[int, float]:
    best_score = -1.0
    best_pos = expected_pos
    for pos in range(expected_pos - radius, expected_pos + radius + 1):
        sl = rx_symbols[pos:pos + len(pilot_ref)]
        if len(sl) != len(pilot_ref):
            continue
        score = pilot_score(sl, pilot_ref)
        if score > best_score:
            best_score = score
            best_pos = pos
    return best_pos, best_score


def estimate_channel(rx_pilot: np.ndarray, tx_pilot: np.ndarray) -> complex:
    return complex(np.vdot(tx_pilot, rx_pilot) / (np.vdot(tx_pilot, tx_pilot) + 1e-12))


def estimate_snr_db(rx_pilot: np.ndarray, tx_pilot: np.ndarray, h_hat: complex) -> float:
    signal = h_hat * tx_pilot
    noise = rx_pilot - signal
    signal_power = float(np.mean(np.abs(signal) ** 2))
    noise_power = float(np.mean(np.abs(noise) ** 2))
    if noise_power <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(max(signal_power, 1e-12) / noise_power))


def evaluate_phase(
    phase: int,
    post_costas: np.ndarray,
    sps: int,
    metadata: dict,
    preamble_ref: np.ndarray,
    pilot_bits: np.ndarray,
    pilot_ref: np.ndarray,
    expected_payload_bits: list[np.ndarray],
    search_start: int,
    coarse_step: int,
    refine_radius: int,
    payload_search_radius: int,
    pilots_to_check: int,
) -> dict:
    rx_samples = np.asarray(post_costas, dtype=np.complex64)
    rx_symbols = averaged_symbols(rx_samples, sps, phase)
    modulation = metadata.get("modulation", "QPSK")
    bps = metadata.get("bits_per_symbol", bits_per_symbol(modulation))
    preamble_symbols = metadata.get("preamble_symbols", metadata.get("preamble_dibits", metadata["preamble_bits"] // bps))
    sync_tail_symbols = int(metadata.get("sync_tail_symbols", min(64, preamble_symbols)))
    pilot_symbols = metadata.get("pilot_symbols", metadata.get("pilot_dibits", metadata["pilot_bits"] // bps))
    data_symbols = metadata.get("data_symbols", metadata.get("data_dibits", metadata["data_bits"] // bps))
    frame_symbols = pilot_symbols + data_symbols
    num_payloads = metadata["num_payloads"]
    effective_search_start = min(search_start, max(0, len(rx_symbols) // 2))

    if num_payloads == 1:
        sync_ref = preamble_ref
        sync_offset_symbols = 0
        sync_differential = False
        if sync_tail_symbols > 0 and sync_tail_symbols < preamble_symbols:
            sync_ref = preamble_ref[-sync_tail_symbols:]
            sync_offset_symbols = preamble_symbols - sync_tail_symbols
            sync_differential = modulation in {"16QAM", "64QAM"}
        preamble_lock = find_frame_start_from_preamble(
            rx_symbols,
            sync_ref,
            search_start=effective_search_start + sync_offset_symbols,
            coarse_step=coarse_step,
            refine_radius=refine_radius,
            differential=sync_differential,
        )
        frame_start = max(0, int(preamble_lock["frame_start"] - sync_offset_symbols))
        sync_start = frame_start + sync_offset_symbols
        omega_hat, phase_hat = estimate_sync_rotation_from_oversampled(
            rx_samples,
            sync_ref,
            sps=sps,
            phase=phase,
            sync_start_symbols=sync_start,
        )
        if omega_hat != 0.0 or phase_hat != 0.0:
            rx_samples = derotate_symbols(rx_samples, omega_hat, phase_hat)
            rx_symbols = averaged_symbols(rx_samples, sps, phase)
            preamble_lock = find_frame_start_from_preamble(
                rx_symbols,
                sync_ref,
                search_start=effective_search_start + sync_offset_symbols,
                coarse_step=coarse_step,
                refine_radius=refine_radius,
                differential=sync_differential,
            )
            frame_start = max(0, int(preamble_lock["frame_start"] - sync_offset_symbols))
            sync_start = frame_start + sync_offset_symbols
        rx_sync = rx_symbols[sync_start:sync_start + len(sync_ref)]
        if len(rx_sync) == len(sync_ref):
            rx_sync = rx_symbols[sync_start:sync_start + len(sync_ref)]
            sync_score = differential_segment_score(rx_sync, sync_ref) if sync_differential else segment_score(rx_sync, sync_ref)
        else:
            sync_score = float(preamble_lock["score"])
        frame_lock = {
            "frame_start": frame_start,
            "pilot_start": int(frame_start + preamble_symbols),
            "score": float(sync_score),
            "checked": 1,
            "sync_source": "sync_tail" if sync_offset_symbols else "preamble",
            "sync_omega_hat": float(omega_hat),
            "sync_phase_hat": float(phase_hat),
        }
    else:
        pilot_lock = find_frame_start(
            rx_symbols,
            preamble_symbols,
            pilot_ref,
            frame_symbols,
            num_payloads,
            search_start=effective_search_start,
            coarse_step=coarse_step,
            refine_radius=refine_radius,
            pilots_to_check=pilots_to_check,
        )
        frame_lock = {
            **pilot_lock,
            "sync_source": "pilot",
            "sync_omega_hat": 0.0,
            "sync_phase_hat": 0.0,
        }

    payload_results = []
    equalized_payload_symbols = []
    equalized_pilot_symbols = []
    detected_payload_bits = []

    for payload_idx in range(num_payloads):
        expected_pilot_pos = frame_lock["pilot_start"] + payload_idx * frame_symbols
        local_search_radius = payload_search_radius
        if frame_lock["sync_source"] == "preamble" and num_payloads == 1 and pilot_symbols <= 2:
            local_search_radius = 0
        pilot_pos, score = nearest_pilot(rx_symbols, pilot_ref, expected_pilot_pos, local_search_radius)
        rx_pilot = rx_symbols[pilot_pos:pilot_pos + pilot_symbols]
        rx_payload = rx_symbols[pilot_pos + pilot_symbols:pilot_pos + pilot_symbols + data_symbols]
        if len(rx_pilot) != pilot_symbols or len(rx_payload) != data_symbols:
            break

        h_hat = estimate_channel(rx_pilot, pilot_ref)
        snr_db = estimate_snr_db(rx_pilot, pilot_ref, h_hat)
        eq_pilot = rx_pilot / (h_hat + 1e-12)
        eq_payload = rx_payload / (h_hat + 1e-12)
        detected_pilot_bits = symbols_to_bits(eq_pilot, modulation)
        detected_bits = symbols_to_bits(eq_payload, modulation)

        equalized_pilot_symbols.append(eq_pilot.astype(np.complex64))
        equalized_payload_symbols.append(eq_payload.astype(np.complex64))
        detected_payload_bits.append(detected_bits)
        payload_results.append(
            {
                "payload_index": payload_idx,
                "pilot_pos": int(pilot_pos),
                "pilot_drift": int(pilot_pos - expected_pilot_pos),
                "pilot_score": float(score),
                "pilot_ber": float(np.mean(detected_pilot_bits != pilot_bits)),
                "channel_real": float(np.real(h_hat)),
                "channel_imag": float(np.imag(h_hat)),
                "channel_mag": float(np.abs(h_hat)),
                "channel_phase_deg": float(np.degrees(np.angle(h_hat))),
                "snr_db": snr_db,
            }
        )

    best_offset = 0
    best_errors = None
    for offset in range(num_payloads):
        total = 0
        for observed_idx, detected_bits in enumerate(detected_payload_bits):
            expected_bits = expected_payload_bits[(offset + observed_idx) % num_payloads]
            total += int(np.sum(detected_bits != expected_bits))
        if best_errors is None or total < best_errors:
            best_errors = total
            best_offset = offset

    total_errors = 0
    total_bits = 0
    mean_pilot_ber = 0.0
    for observed_idx, result in enumerate(payload_results):
        expected_idx = (best_offset + observed_idx) % num_payloads
        expected_bits = expected_payload_bits[expected_idx]
        detected_bits = detected_payload_bits[observed_idx]
        errors = int(np.sum(detected_bits != expected_bits))
        ber = errors / len(expected_bits)
        result["expected_payload_index"] = expected_idx
        result["bit_errors"] = errors
        result["bits"] = int(len(expected_bits))
        result["ber"] = float(ber)
        total_errors += errors
        total_bits += len(expected_bits)
        mean_pilot_ber += result["pilot_ber"]
    mean_pilot_ber /= max(1, len(payload_results))
    mean_snr_db = float(np.mean([item["snr_db"] for item in payload_results])) if payload_results else float("nan")

    return {
        "phase": phase,
        "frame_start": frame_lock["frame_start"],
        "pilot_start": frame_lock["pilot_start"],
        "lock_score": frame_lock["score"],
        "frame_lock_pilots_checked": frame_lock["checked"],
        "sync_source": frame_lock["sync_source"],
        "sync_omega_hat": frame_lock["sync_omega_hat"],
        "sync_phase_hat": frame_lock["sync_phase_hat"],
        "payload_cycle_offset": best_offset,
        "mean_pilot_ber": mean_pilot_ber,
        "mean_snr_db": mean_snr_db,
        "overall_ber": (total_errors / total_bits) if total_bits else 1.0,
        "total_errors": int(total_errors),
        "total_bits": int(total_bits),
        "payload_results": payload_results,
        "equalized_payload_symbols": np.array(equalized_payload_symbols, dtype=np.complex64),
        "equalized_pilot_symbols": np.array(equalized_pilot_symbols, dtype=np.complex64),
        "decimated_symbols": rx_symbols.astype(np.complex64),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate oversampled capture from post-recovery samples.")
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--post-costas-file", default="received_post_costas_qpsk.bin")
    parser.add_argument("--tx-frame", default="transmitted_frame.bin")
    parser.add_argument("--metadata", default="frame_metadata.json")
    parser.add_argument("--output-prefix", default="qpsk_post_costas_validation")
    parser.add_argument("--sps", type=int, default=25)
    parser.add_argument("--search-start", type=int, default=200000)
    parser.add_argument("--coarse-step", type=int, default=128)
    parser.add_argument("--refine-radius", type=int, default=64)
    parser.add_argument("--payload-search-radius", type=int, default=12)
    parser.add_argument("--pilots-to-check", type=int, default=4)
    parser.add_argument("--phase-start", type=int, default=0)
    parser.add_argument("--phase-stop", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    capture_dir = Path(args.capture_dir)
    post_costas_path = capture_dir / args.post_costas_file
    tx_frame_path = capture_dir / args.tx_frame
    metadata_path = capture_dir / args.metadata

    metadata = load_metadata(metadata_path)
    post_costas = np.fromfile(post_costas_path, dtype=np.complex64)
    tx_bits = np.fromfile(tx_frame_path, dtype=np.uint8)

    modulation = metadata.get("modulation", "QPSK")
    preamble_bits = tx_bits[:metadata["preamble_bits"]]
    preamble_ref = bits_to_symbols(preamble_bits, modulation)
    pilot_bits = tx_bits[metadata["preamble_bits"]:metadata["preamble_bits"] + metadata["pilot_bits"]]
    pilot_ref = bits_to_symbols(pilot_bits, modulation)
    expected_payload_bits = load_expected_payloads(metadata, capture_dir, tx_bits)

    phase_summaries = []
    best = None
    for phase in range(args.phase_start, min(args.phase_stop, args.sps)):
        result = evaluate_phase(
            phase=phase,
            post_costas=post_costas,
            sps=args.sps,
            metadata=metadata,
            preamble_ref=preamble_ref,
            pilot_bits=pilot_bits,
            pilot_ref=pilot_ref,
            expected_payload_bits=expected_payload_bits,
            search_start=args.search_start,
            coarse_step=args.coarse_step,
            refine_radius=args.refine_radius,
            payload_search_radius=args.payload_search_radius,
            pilots_to_check=args.pilots_to_check,
        )
        phase_summaries.append(
            {
                "phase": result["phase"],
                "lock_score": result["lock_score"],
                "mean_pilot_ber": result["mean_pilot_ber"],
                "overall_ber": result["overall_ber"],
                "payload_cycle_offset": result["payload_cycle_offset"],
            }
        )
        sort_key = (-result["lock_score"], result["overall_ber"], result["mean_pilot_ber"])
        if best is None or sort_key < best["sort_key"]:
            best = {"sort_key": sort_key, "result": result}

    result = best["result"]
    output_prefix = capture_dir / args.output_prefix
    summary_path = output_prefix.with_suffix(".json")
    npz_path = output_prefix.with_suffix(".npz")

    summary = {
        "capture_dir": str(capture_dir),
        "post_costas_file": str(post_costas_path),
        "tx_frame": str(tx_frame_path),
        "metadata": str(metadata_path),
        "best_phase": {
            "phase": result["phase"],
            "frame_start": result["frame_start"],
            "pilot_start": result["pilot_start"],
            "lock_score": result["lock_score"],
                "frame_lock_pilots_checked": result["frame_lock_pilots_checked"],
                "sync_source": result["sync_source"],
                "payload_cycle_offset": result["payload_cycle_offset"],
                "mean_pilot_ber": result["mean_pilot_ber"],
                "mean_snr_db": result["mean_snr_db"],
                "overall_ber": result["overall_ber"],
            "total_errors": result["total_errors"],
            "total_bits": result["total_bits"],
            "payload_results": result["payload_results"],
        },
        "all_phases": phase_summaries,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez(
        npz_path,
        decimated_symbols=result["decimated_symbols"],
        equalized_payload_symbols=result["equalized_payload_symbols"],
        equalized_pilot_symbols=result["equalized_pilot_symbols"],
    )

    print("=" * 72)
    print(f"{modulation.upper()} POST-RECOVERY VALIDATION")
    print("=" * 72)
    print(f"Capture dir:          {capture_dir}")
    print(f"Post-Costas file:     {post_costas_path}")
    print(f"Best phase:           {result['phase']}")
    print(f"Frame start:          {result['frame_start']}")
    print(f"First pilot:          {result['pilot_start']}")
    print(f"Lock score:           {result['lock_score']:.4f}")
    print(f"Sync source:          {result['sync_source']}")
    print(f"Payload cycle offset: {result['payload_cycle_offset']}")
    print(f"Mean pilot BER:       {result['mean_pilot_ber']:.4f}")
    print(f"Mean SNR:             {result['mean_snr_db']:.2f} dB")
    print(f"Overall BER:          {result['overall_ber']:.6f} ({result['total_errors']}/{result['total_bits']})")
    print(f"Summary JSON:         {summary_path}")
    print(f"Best-phase symbols:   {npz_path}")
    print()
    for item in result["payload_results"]:
        print(
            f"  Payload {item['payload_index']:02d}: "
            f"exp={item['expected_payload_index']:02d} "
            f"pilotBER={item['pilot_ber']:.4f} "
            f"SNR={item['snr_db']:.2f}dB "
            f"BER={item['ber']:.4f} "
            f"score={item['pilot_score']:.4f} "
            f"drift={item['pilot_drift']:+d}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
