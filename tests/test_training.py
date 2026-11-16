"""Training smoke tests for the FNO-2D thermal surrogate.

Runs a single optimizer step on a tiny generated dataset to confirm gradients
flow and weights update.
"""

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from thermal_surrogate_ml.data.dataset import ThermalDataset
from thermal_surrogate_ml.data.generate import generate_dataset
from thermal_surrogate_ml.models.fno import FNO2d
from thermal_surrogate_ml.training.train import get_device, relative_l2


@pytest.fixture(scope="module")
def tiny_paths(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Generate one tiny dataset shared across this test module."""
    out_dir = tmp_path_factory.mktemp("training-data")
    return generate_dataset(
        out_dir=out_dir,
        n_train=4,
        n_val=2,
        n_test=2,
        grid_size=16,
        length=1.0,
        alpha=1.0,
        mode="transient",
        t_final=0.02,
        seed=0,
    )


def test_get_device() -> None:
    device = get_device()
    assert device.type in {"cuda", "cpu"}


def test_single_training_step(tiny_paths: dict[str, Path]) -> None:
    torch.manual_seed(0)

    train_ds = ThermalDataset(
        tiny_paths["train"], include_initial=False, normalize=True
    )
    loader = DataLoader(train_ds, batch_size=2, shuffle=False)

    # modes2 must not exceed (grid_size // 2 + 1) for the 16x16 field.
    model = FNO2d(modes1=8, modes2=8, width=16, in_channels=1, out_channels=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    x, y = next(iter(loader))
    before = {name: p.detach().clone() for name, p in model.named_parameters()}

    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)

    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    assert any(
        not torch.equal(before[name], param.detach())
        for name, param in model.named_parameters()
    )


def test_relative_l2_is_normalized() -> None:
    target = torch.ones(2, 1, 4, 4)

    perfect = target.clone()
    assert float(relative_l2(perfect, target)) == pytest.approx(0.0, abs=1e-6)

    zero = torch.zeros_like(target)
    assert float(relative_l2(zero, target)) == pytest.approx(1.0, abs=1e-4)