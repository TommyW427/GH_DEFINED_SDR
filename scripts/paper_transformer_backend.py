#!/usr/bin/env python3
"""
Checkpoint-loaded paper-style Transformer backends for ICL and DEFINED.

This module assumes trained model weights already exist. It does not train.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.modulation import constellation_points, normalize_modulation


def _lazy_torch_import():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Paper Transformer backends require a working torch installation in the active interpreter."
        ) from exc
    return torch, nn, F


def _checkpoint_env_name(method: str) -> str:
    return f"PAPER_{method.upper()}_CHECKPOINT"


def _default_checkpoint_path(method: str, modulation: str) -> Path:
    method_tag = method.upper()
    modulation_tag = normalize_modulation(modulation).upper()
    candidates = [
        Path(f"trained_model_{method_tag}_{modulation_tag}.pth"),
        Path("models") / f"trained_model_{method_tag}_{modulation_tag}.pth",
        Path("models") / f"paper_{method.lower()}_{modulation_tag.lower()}.pt",
        Path("models") / f"paper_{method.lower()}_{modulation_tag.lower()}.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_checkpoint_path(method: str, modulation: str) -> Path:
    env_path = os.environ.get(_checkpoint_env_name(method))
    candidate = Path(env_path) if env_path else _default_checkpoint_path(method, modulation)
    if not candidate.exists():
        raise FileNotFoundError(
            f"Missing checkpoint for {method}/{modulation}: {candidate}. "
            f"Set {_checkpoint_env_name(method)} or place a checkpoint at the default path."
        )
    return candidate


def _class_permutation_env_name(modulation: str) -> str:
    return f"PAPER_CLASS_PERM_{normalize_modulation(modulation).upper()}"


def _parse_class_permutation(value: str, expected: int) -> list[int]:
    parts = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
    perm = [int(part) for part in parts]
    if sorted(perm) != list(range(expected)):
        raise ValueError(f"Invalid class permutation {value!r}; expected a permutation of 0..{expected - 1}.")
    return perm


def _default_class_permutation(modulation: str, num_classes: int) -> list[int]:
    modulation = normalize_modulation(modulation)
    env_value = os.environ.get(_class_permutation_env_name(modulation)) or os.environ.get("PAPER_CLASS_PERM")
    if env_value:
        return _parse_class_permutation(env_value, num_classes)
    if modulation == "QPSK" and num_classes == 4:
        # The supplied QPSK checkpoints use the natural Cartesian class order:
        # [++, +-, -+, --]. Our modem's Gray bit order is [++, -+, +-, --].
        return [0, 2, 1, 3]
    return list(range(num_classes))


@dataclass
class RuntimeConfig:
    method: str
    modulation: str
    total_symbols: int = 31
    d_model: int = 64
    nhead: int = 4
    n_layers: int = 4
    feature_dim: int = 3
    num_classes: int = 4


class PromptTransformer:
    def __init__(
        self,
        torch,
        nn,
        feature_dim: int,
        num_classes: int,
        d_model: int,
        nhead: int,
        n_layers: int,
        dim_feedforward: int = 2048,
    ):
        self._torch = torch
        self._nn = nn
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.model = self._build(d_model, nhead, n_layers, dim_feedforward)

    def _build(self, d_model: int, nhead: int, n_layers: int, dim_feedforward: int):
        nn = self._nn

        class _Model(nn.Module):
            def __init__(
                self,
                feature_dim: int,
                num_classes: int,
                d_model: int,
                nhead: int,
                n_layers: int,
                dim_feedforward: int,
            ):
                super().__init__()
                self.inp = nn.Linear(feature_dim, d_model)
                enc = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    batch_first=True,
                )
                self.tr = nn.TransformerEncoder(enc, num_layers=n_layers)
                self.out = nn.Linear(d_model, num_classes)

            def forward(self, x):
                x = self.inp(x)
                x = self.tr(x)
                return self.out(x)

        return _Model(self.feature_dim, self.num_classes, d_model, nhead, n_layers, dim_feedforward)


class PaperTransformerRuntime:
    def __init__(self, method: str, modulation: str, total_symbols: int = 31):
        torch, nn, F = _lazy_torch_import()
        self.torch = torch
        self.nn = nn
        self.F = F
        self.method = method
        self.modulation = normalize_modulation(modulation)
        self.total_symbols = int(total_symbols)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.runtime_config = self._load_runtime_config()
        self.constellation = np.array(constellation_points(self.modulation), dtype=np.complex64)
        if len(self.constellation) != self.runtime_config.num_classes:
            raise ValueError(
                f"Checkpoint class count ({self.runtime_config.num_classes}) does not match "
                f"{self.modulation} constellation size ({len(self.constellation)})."
            )
        self.num_classes = self.runtime_config.num_classes
        self.class_permutation = _default_class_permutation(self.modulation, self.num_classes)
        self.constellation = self.constellation[self.class_permutation]
        self.feature_dim = self.runtime_config.feature_dim
        self.network = PromptTransformer(
            torch,
            nn,
            feature_dim=self.feature_dim,
            num_classes=self.num_classes,
            d_model=self.runtime_config.d_model,
            nhead=self.runtime_config.nhead,
            n_layers=self.runtime_config.n_layers,
        )
        self.model = self.network.model.to(self.device)
        self._load_weights()
        self.model.eval()

    def _load_runtime_config(self) -> RuntimeConfig:
        checkpoint_path = _resolve_checkpoint_path(self.method, self.modulation)
        torch = self.torch
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
        state_dict = self._extract_state_dict(checkpoint)
        feature_dim = int(state_dict["inp.weight"].shape[1])
        d_model = int(state_dict["inp.weight"].shape[0])
        num_classes = int(state_dict["out.weight"].shape[0])
        layer_indices = [
            int(key.split(".")[2])
            for key in state_dict
            if key.startswith("tr.layers.") and len(key.split(".")) > 2 and key.split(".")[2].isdigit()
        ]
        n_layers = max(layer_indices) + 1 if layer_indices else int(cfg.get("n_layers", 4))
        dim_feedforward = int(state_dict.get("tr.layers.0.linear1.weight").shape[0]) if "tr.layers.0.linear1.weight" in state_dict else 2048
        nhead = int(cfg.get("nhead", 4))
        return RuntimeConfig(
            method=self.method,
            modulation=cfg.get("modulation", self.modulation),
            total_symbols=int(cfg.get("total_symbols", cfg.get("T", self.total_symbols))),
            d_model=int(cfg.get("d_model", d_model)),
            nhead=nhead,
            n_layers=int(cfg.get("n_layers", n_layers)),
            feature_dim=feature_dim,
            num_classes=num_classes,
        )

    def _extract_state_dict(self, checkpoint):
        if isinstance(checkpoint, (dict, OrderedDict)):
            return checkpoint.get("state_dict") or checkpoint.get("model_state_dict") or checkpoint
        return checkpoint

    def _load_weights(self) -> None:
        checkpoint_path = _resolve_checkpoint_path(self.method, self.modulation)
        torch = self.torch
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = self._extract_state_dict(checkpoint)
        load_result = self.model.load_state_dict(state_dict, strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            print(
                f"[{self.method}/{self.modulation}] checkpoint loaded with "
                f"missing={load_result.missing_keys} unexpected={load_result.unexpected_keys}"
            )

    def _symbol_to_index(self, symbol: complex) -> int:
        diffs = np.abs(self.constellation - np.complex64(symbol))
        return int(np.argmin(diffs))

    def _build_prompt(self, context, query_symbol: complex):
        torch = self.torch
        k = len(context)
        feat = torch.zeros(1, k + 1, self.feature_dim, device=self.device)
        # PaperICL-2.py and PaperDEFINED-2.py use scalar class indices in
        # feature 2, not one-hot or one-based labels.
        label_encoding = os.environ.get("PAPER_LABEL_ENCODING", "zero_based").lower()
        for idx, pair in enumerate(context):
            feat[0, idx, 0] = float(np.real(pair.received))
            feat[0, idx, 1] = float(np.imag(pair.received))
            class_idx = self._symbol_to_index(pair.transmitted)
            if self.feature_dim == 3:
                if label_encoding in {"zero_based", "class_index"}:
                    label_value = float(class_idx)
                elif label_encoding in {"normalized", "unit"}:
                    label_value = float(class_idx / max(1, self.num_classes - 1))
                else:
                    label_value = float(class_idx + 1)
                feat[0, idx, 2] = label_value
            else:
                feat[0, idx, 2 + class_idx] = 1.0
        feat[0, k, 0] = float(np.real(query_symbol))
        feat[0, k, 1] = float(np.imag(query_symbol))
        if self.feature_dim == 3:
            feat[0, k, 2] = float(os.environ.get("PAPER_QUERY_LABEL", "0"))
        return feat

    def detect_symbol_with_context(self, context, query_symbol: complex) -> complex:
        if not context:
            raise ValueError("Paper Transformer backends require at least one context pair.")
        with self.torch.no_grad():
            prompt = self._build_prompt(context, query_symbol)
            logits = self.model(prompt)[:, -1, :]
            pred_idx = int(self.torch.argmax(logits, dim=-1).item())
        return complex(self.constellation[pred_idx])


_BACKEND_CACHE: dict[tuple[str, str, int], PaperTransformerRuntime] = {}


def get_runtime(method: str, modulation: str, total_symbols: int = 31):
    key = (method, normalize_modulation(modulation), int(total_symbols))
    runtime = _BACKEND_CACHE.get(key)
    if runtime is None:
        runtime = PaperTransformerRuntime(method=method, modulation=modulation, total_symbols=total_symbols)
        _BACKEND_CACHE[key] = runtime
    return runtime
