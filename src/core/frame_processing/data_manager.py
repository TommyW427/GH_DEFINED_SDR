#!/usr/bin/env python3
"""
Data Manager for Frame Construction
====================================

Manages preamble, pilot, and payload data for frame construction.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .config import FrameConfig


class DataManager:
    """
    Manages frame data: preamble, pilots, and payloads.

    Handles:
    - Generating or loading preamble and pilot sequences
    - Generating random payloads
    - Building complete frames: [preamble][pilot][data][pilot][data]...
    - Saving/loading frames and metadata
    """

    def __init__(self, config: FrameConfig, preamble_file: Optional[str] = None, pilot_file: Optional[str] = None):
        """
        Initialize DataManager.

        Args:
            config: Frame configuration
            preamble_file: Path to preamble file (if None, generates new one)
            pilot_file: Path to pilot file (if None, generates new one)
        """
        self.config = config
        self.preamble_bits = config.preamble_bits
        self.pilot_bits = config.pilot_bits
        self.data_bits = config.data_bits

        if preamble_file is None:
            self.preamble = self._generate_preamble()
        else:
            self.preamble = np.fromfile(preamble_file, dtype=np.uint8)

        if pilot_file is None:
            self.pilot = self._generate_pilot()
        else:
            self.pilot = np.fromfile(pilot_file, dtype=np.uint8)

    def _generate_preamble(self, seed: int = 42) -> np.ndarray:
        """
        Generate preamble sequence.

        Uses a fixed seed for reproducibility across TX/RX.

        Args:
            seed: Random seed

        Returns:
            Preamble bit array
        """
        rng = np.random.default_rng(seed=seed)
        preamble = rng.integers(0, 2, self.preamble_bits, dtype=np.uint8)
        return preamble

    def _generate_pilot(self, seed: int = 123) -> np.ndarray:
        """
        Generate pilot sequence.

        Uses a fixed seed for reproducibility across TX/RX.

        Args:
            seed: Random seed

        Returns:
            Pilot bit array
        """
        rng = np.random.default_rng(seed=seed)
        pilot = rng.integers(0, 2, self.pilot_bits, dtype=np.uint8)
        return pilot

    def generate_random_payload(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate random payload data.

        Args:
            seed: Optional random seed (None = random)

        Returns:
            Payload bit array
        """
        rng = np.random.default_rng(seed=seed)
        payload = rng.integers(0, 2, self.data_bits, dtype=np.uint8)
        return payload

    def build_frame(self, payload_files: List[str]) -> Tuple[np.ndarray, Dict]:
        """
        Build complete frame from payload files.

        Frame structure: [preamble][pilot][data0][pilot][data1]...[pilot][dataN]

        Args:
            payload_files: List of paths to payload files

        Returns:
            Tuple of (frame_bits, metadata_dict)
        """
        frame_parts = [self.preamble]

        for payload_file in payload_files:
            payload = np.fromfile(payload_file, dtype=np.uint8)
            frame_parts.append(self.pilot)
            frame_parts.append(payload)

        frame = np.concatenate(frame_parts)

        metadata = {
            'preamble_bits': self.preamble_bits,
            'pilot_bits': self.pilot_bits,
            'data_bits': self.data_bits,
            'num_payloads': len(payload_files),
            'payload_files': payload_files,
            'frame_length': len(frame)
        }

        return frame, metadata

    def save_frame(self, frame: np.ndarray, metadata: Dict, output_file: str, metadata_file: str):
        """
        Save frame and metadata to disk.

        Args:
            frame: Frame bit array
            metadata: Metadata dictionary
            output_file: Path to save frame bits
            metadata_file: Path to save JSON metadata
        """
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        frame.tofile(output_file)

        # Convert numpy types to native Python for JSON serialization
        json_metadata = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in metadata.items()}

        with open(metadata_file, 'w') as f:
            json.dump(json_metadata, f, indent=2)

        print(f"[DataManager] Frame saved:")
        print(f"  Frame: {output_file} ({len(frame)} bits)")
        print(f"  Metadata: {metadata_file}")

    def load_frame_metadata(self, metadata_file: str) -> Dict:
        """
        Load frame metadata from JSON file.

        Args:
            metadata_file: Path to metadata file

        Returns:
            Metadata dictionary
        """
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        return metadata

    def save_preamble(self, filepath: str):
        """Save preamble to file."""
        self.preamble.tofile(filepath)
        print(f"[DataManager] Preamble saved to {filepath}")

    def save_pilots(self, filepath: str):
        """Save pilot sequence to file."""
        self.pilot.tofile(filepath)
        print(f"[DataManager] Pilot saved to {filepath}")

    def generate_payload_dataset(self, num_samples: int, output_dir: str):
        """
        Generate a dataset of random payloads.

        Args:
            num_samples: Number of payloads to generate
            output_dir: Directory to save payloads
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for i in range(num_samples):
            payload = self.generate_random_payload()
            output_file = Path(output_dir) / f"payload_{i:04d}.bin"
            payload.tofile(output_file)

        print(f"[DataManager] Generated {num_samples} payloads in {output_dir}")
