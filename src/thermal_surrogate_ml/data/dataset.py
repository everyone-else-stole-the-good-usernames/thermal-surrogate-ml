"""PyTorch ``Dataset`` for the thermal surrogate.

Items are ``torch.float32`` tensors: input ``[C, H, W]`` (``C = 1`` for the
source ``Q``, or ``2`` when the initial temperature is included) and target
``[1, H, W]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class NormalizationStats:
    """Per-channel input statistics and scalar target statistics."""

    input_mean: np.ndarray  # [C]
    input_std: np.ndarray  # [C]
    input_min: np.ndarray  # [C]
    input_max: np.ndarray  # [C]
    target_mean: float
    target_std: float
    target_min: float
    target_max: float


def compute_stats(
    path: str | Path,
    include_initial: bool = False,
) -> NormalizationStats:
    """Compute normalization statistics over a single ``.npz`` split."""

    with np.load(path) as data:
        source = np.asarray(data["source"], dtype=np.float64)
        initial = np.asarray(data["initial"], dtype=np.float64)
        target = np.asarray(data["target"], dtype=np.float64)

    if include_initial:
        inputs = np.stack([source, initial], axis=1)  # [N, C, H, W]
    else:
        inputs = source[:, None, :, :]  # [N, 1, H, W]

    return NormalizationStats(
        input_mean=inputs.mean(axis=(0, 2, 3)),
        input_std=inputs.std(axis=(0, 2, 3)),
        input_min=inputs.min(axis=(0, 2, 3)),
        input_max=inputs.max(axis=(0, 2, 3)),
        target_mean=float(target.mean()),
        target_std=float(target.std()),
        target_min=float(target.min()),
        target_max=float(target.max()),
    )


class ThermalDataset(Dataset):
    """Loads a generated thermal ``.npz`` and yields normalized tensors."""

    def __init__(
        self,
        path: str | Path,
        include_initial: bool = False,
        normalize: bool = True,
        norm: Literal["zscore", "minmax"] = "zscore",
        stats: NormalizationStats | None = None,
    ) -> None:
        if norm not in ("zscore", "minmax"):
            raise ValueError(f"norm must be 'zscore' or 'minmax', got {norm!r}")

        self.path = Path(path)
        self.include_initial = include_initial
        self.normalize = normalize
        self.norm = norm

        with np.load(self.path) as data:
            source = np.asarray(data["source"], dtype=np.float32)
            initial = np.asarray(data["initial"], dtype=np.float32)
            target = np.asarray(data["target"], dtype=np.float32)

        if include_initial:
            self.inputs = np.stack([source, initial], axis=1)  # [N, C, H, W]
        else:
            self.inputs = source[:, None, :, :]
        self.targets = target[:, None, :, :]  # [N, 1, H, W]

        if self.normalize and stats is None:
            stats = compute_stats(self.path, include_initial=include_initial)
        self.stats = stats

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.inputs[index].astype(np.float32, copy=False)
        y = self.targets[index].astype(np.float32, copy=False)

        if self.normalize and self.stats is not None:
            x = self._normalize_input(x)
            y = self._normalize_target(y)

        return (
            torch.from_numpy(np.ascontiguousarray(x)),
            torch.from_numpy(np.ascontiguousarray(y)),
        )

    def _normalize_input(self, x: np.ndarray) -> np.ndarray:
        stats = self.stats
        assert stats is not None

        if self.norm == "zscore":
            mean = stats.input_mean[:, None, None]
            std = stats.input_std[:, None, None] + 1e-8
            return ((x - mean) / std).astype(np.float32)

        low = stats.input_min[:, None, None]
        scale = (stats.input_max - stats.input_min + 1e-8)[:, None, None]
        return ((x - low) / scale).astype(np.float32)

    def _normalize_target(self, y: np.ndarray) -> np.ndarray:
        stats = self.stats
        assert stats is not None

        if self.norm == "zscore":
            return ((y - stats.target_mean) / (stats.target_std + 1e-8)).astype(
                np.float32
            )
        return (
            (y - stats.target_min) / (stats.target_max - stats.target_min + 1e-8)
        ).astype(np.float32)