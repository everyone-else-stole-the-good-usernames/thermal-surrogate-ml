"""Unit tests for the 2D Fourier Neural Operator."""

import pytest
import torch

from thermal_surrogate_ml.models.fno import FNO2d


@pytest.fixture(scope="module")
def fno() -> FNO2d:
    torch.manual_seed(0)
    return FNO2d(modes1=12, modes2=12, width=32)


def test_forward_shape_64x64(fno: FNO2d) -> None:
    x = torch.randn(8, 1, 64, 64)

    with torch.no_grad():
        y = fno(x)

    assert y.shape == (8, 1, 64, 64)
    assert y.dtype == torch.float32


def test_zero_shot_super_resolution_128x128(fno: FNO2d) -> None:
    x = torch.randn(2, 1, 128, 128)

    with torch.no_grad():
        y = fno(x)

    assert y.shape == (2, 1, 128, 128)
    assert y.dtype == torch.float32
    assert torch.isfinite(y).all()