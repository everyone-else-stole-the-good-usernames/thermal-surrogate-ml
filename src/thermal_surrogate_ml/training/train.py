"""Training loop for the FNO-2D thermal surrogate.

Device-agnostic (CUDA or CPU; MPS is excluded to avoid PyTorch FFT backend
pitfalls on Apple Silicon). Metrics are tracked in z-scored space for
optimization stability; physical-space metrics come from
:mod:`thermal_surrogate_ml.training.evaluate`.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from thermal_surrogate_ml.data.dataset import NormalizationStats, ThermalDataset
from thermal_surrogate_ml.models.fno import FNO2d

_EPS = 1e-8


def get_device() -> torch.device:
    """Resolve the compute device (``cuda`` if available, else ``cpu``)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def relative_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Average per-sample relative L2 error over a batch.

    Reduces over ``(1, 2, 3)`` so the result is batch-size independent; epsilon
    guards against a vanishing target norm.
    """
    numerator = torch.linalg.vector_norm(pred - target, dim=(1, 2, 3))
    denominator = torch.linalg.vector_norm(target, dim=(1, 2, 3)) + _EPS
    return (numerator / denominator).mean()


def stats_to_dict(stats: NormalizationStats) -> dict[str, object]:
    """Serialize :class:`NormalizationStats` for checkpoint persistence."""
    return {
        "input_mean": stats.input_mean,
        "input_std": stats.input_std,
        "input_min": stats.input_min,
        "input_max": stats.input_max,
        "target_mean": stats.target_mean,
        "target_std": stats.target_std,
        "target_min": stats.target_min,
        "target_max": stats.target_max,
    }


def stats_from_dict(payload: dict[str, object]) -> NormalizationStats:
    """Reconstruct :class:`NormalizationStats` from a serialized dictionary."""
    return NormalizationStats(
        input_mean=np.asarray(payload["input_mean"]),
        input_std=np.asarray(payload["input_std"]),
        input_min=np.asarray(payload["input_min"]),
        input_max=np.asarray(payload["input_max"]),
        target_mean=float(payload["target_mean"]),
        target_std=float(payload["target_std"]),
        target_min=float(payload["target_min"]),
        target_max=float(payload["target_max"]),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch, returning ``(mean MSE, mean relative L2)``."""
    model.train()
    total_loss = 0.0
    total_rel = 0.0
    num_samples = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        batch = x.size(0)
        total_loss += float(loss.detach()) * batch
        total_rel += float(relative_l2(pred.detach(), y)) * batch
        num_samples += batch

    return total_loss / num_samples, total_rel / num_samples


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the model, returning ``(mean MSE, mean relative L2)``."""
    model.eval()
    total_loss = 0.0
    total_rel = 0.0
    num_samples = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x)
        loss = criterion(pred, y)

        batch = x.size(0)
        total_loss += float(loss) * batch
        total_rel += float(relative_l2(pred, y)) * batch
        num_samples += batch

    return total_loss / num_samples, total_rel / num_samples


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for training."""
    parser = argparse.ArgumentParser(
        description="Train the FNO-2D thermal surrogate."
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--scheduler", choices=["cosine", "step"], default="cosine"
    )
    parser.add_argument("--step-size", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=0.5)
    args = parser.parse_args(argv)

    device = get_device()
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.npz"
    val_path = data_dir / "val.npz"
    if not train_path.exists():
        raise FileNotFoundError(f"Training split not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation split not found: {val_path}")

    # Fit normalization on the training split and reuse it for validation so
    # both live in the same normalized space.
    train_ds = ThermalDataset(train_path, include_initial=False, normalize=True)
    train_stats = train_ds.stats
    if train_stats is None:
        raise RuntimeError("Failed to compute training normalization stats.")
    val_ds = ThermalDataset(
        val_path, include_initial=False, normalize=True, stats=train_stats
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = FNO2d(
        modes1=12, modes2=12, width=32, in_channels=1, out_channels=1
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.scheduler == "cosine":
        scheduler: torch.optim.lr_scheduler.LRScheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs
            )
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma
        )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "best_model.pt"

    hyperparameters = {
        "modes1": 12,
        "modes2": 12,
        "width": 32,
        "in_channels": 1,
        "out_channels": 1,
        "include_initial": False,
        "norm": "zscore",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "scheduler": args.scheduler,
    }

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_rel = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_rel = validate(model, val_loader, criterion, device)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "stats": stats_to_dict(train_stats),
                    "hyperparameters": hyperparameters,
                },
                best_path,
            )

        print(
            f"epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={train_loss:.6f} train_rel_l2={train_rel:.6f} | "
            f"val_loss={val_loss:.6f} val_rel_l2={val_rel:.6f}"
        )

    print(f"Best validation loss: {best_val_loss:.6f} -> {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())