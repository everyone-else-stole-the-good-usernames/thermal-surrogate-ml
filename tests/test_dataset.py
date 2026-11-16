"""Unit tests for dataset generation and the PyTorch data pipeline."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from thermal_surrogate_ml.data.dataset import ThermalDataset, compute_stats
from thermal_surrogate_ml.data.generate import generate_dataset


@pytest.fixture(scope="module")
def tiny_dataset_paths(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Generate a tiny dataset once and share it across this test module."""

    out_dir = tmp_path_factory.mktemp("thermal-data")
    return generate_dataset(
        out_dir=out_dir,
        n_train=5,
        n_val=3,
        n_test=3,
        grid_size=32,
        length=1.0,
        alpha=1.0,
        mode="transient",
        t_final=0.02,
        seed=0,
    )


def test_generate_tiny_dataset(tiny_dataset_paths: dict[str, Path]) -> None:
    expected = {"train": 5, "val": 3, "test": 3}

    assert set(tiny_dataset_paths) == {"train", "val", "test"}

    for split, path in tiny_dataset_paths.items():
        assert path.exists()
        with np.load(path) as data:
            assert set(data.files) == {"source", "initial", "target"}
            shape = (expected[split], 32, 32)
            assert data["source"].shape == shape
            assert data["initial"].shape == shape
            assert data["target"].shape == shape
            assert data["source"].dtype == np.float32
            assert data["target"].dtype == np.float32


def test_dataset_returns_required_shapes(tiny_dataset_paths: dict[str, Path]) -> None:
    ds = ThermalDataset(
        tiny_dataset_paths["train"], include_initial=False, normalize=False
    )
    assert len(ds) == 5

    x, y = ds[0]
    assert x.shape == (1, 32, 32)
    assert y.shape == (1, 32, 32)
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32

    ds2 = ThermalDataset(
        tiny_dataset_paths["train"], include_initial=True, normalize=False
    )
    x2, _ = ds2[0]
    assert x2.shape == (2, 32, 32)


def test_dataloader_batching(tiny_dataset_paths: dict[str, Path]) -> None:
    ds = ThermalDataset(
        tiny_dataset_paths["train"], include_initial=False, normalize=False
    )
    loader = DataLoader(ds, batch_size=2, shuffle=False)

    x, y = next(iter(loader))
    assert x.shape == (2, 1, 32, 32)
    assert y.shape == (2, 1, 32, 32)


def test_normalization_reuses_train_stats(tiny_dataset_paths: dict[str, Path]) -> None:
    train_stats = compute_stats(tiny_dataset_paths["train"], include_initial=False)

    train_ds = ThermalDataset(tiny_dataset_paths["train"], stats=train_stats)
    val_ds = ThermalDataset(tiny_dataset_paths["val"], stats=train_stats)

    assert val_ds.stats is train_stats

    targets = np.concatenate([train_ds[i][1].numpy() for i in range(len(train_ds))])

    # Z-scored training targets should have ~zero mean and ~unit standard
    # deviation (up to floating-point and eps regularisation).
    assert abs(float(targets.mean())) < 1e-4
    assert abs(float(targets.std()) - 1.0) < 1e-3


def test_splits_have_no_leakage(tiny_dataset_paths: dict[str, Path]) -> None:
    with np.load(tiny_dataset_paths["train"]) as data:
        train = np.asarray(data["source"])
    with np.load(tiny_dataset_paths["val"]) as data:
        val = np.asarray(data["source"])
    with np.load(tiny_dataset_paths["test"]) as data:
        test = np.asarray(data["source"])

    train_fp = {sample.tobytes() for sample in train.reshape(train.shape[0], -1)}
    val_fp = {sample.tobytes() for sample in val.reshape(val.shape[0], -1)}

    for sample in val.reshape(val.shape[0], -1):
        assert sample.tobytes() not in train_fp
    for sample in test.reshape(test.shape[0], -1):
        assert sample.tobytes() not in train_fp
        assert sample.tobytes() not in val_fp