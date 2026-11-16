"""Evaluation and visualization for the FNO-2D thermal surrogate.

Loads a checkpoint (with its embedded normalization statistics), reports RMSE
and relative L2 in physical units over the test split, and writes a
side-by-side comparison grid.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from thermal_surrogate_ml.data.dataset import NormalizationStats, ThermalDataset
from thermal_surrogate_ml.models.fno import FNO2d
from thermal_surrogate_ml.training.train import get_device, stats_from_dict

_EPS = 1e-8


def denormalize_target(y: torch.Tensor, stats: NormalizationStats) -> torch.Tensor:
    """Invert the target z-score transform: ``y * std + mean``."""
    return y * (stats.target_std + _EPS) + stats.target_mean


def denormalize_input(x: torch.Tensor, stats: NormalizationStats) -> torch.Tensor:
    """Invert the per-channel input z-score transform.

    Statistics broadcast over the channel axis of ``[N, C, H, W]``.
    """
    mean = torch.as_tensor(stats.input_mean, dtype=x.dtype, device=x.device)
    std = torch.as_tensor(stats.input_std, dtype=x.dtype, device=x.device) + _EPS
    return x * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)


def aggregate_metrics(
    y_pred: torch.Tensor, y_true: torch.Tensor
) -> tuple[float, float]:
    """Return aggregate ``(RMSE, relative L2)`` over the full test set."""
    rmse = float(torch.sqrt(torch.mean((y_pred - y_true) ** 2)))

    residual_norm = torch.linalg.vector_norm(y_pred - y_true)
    target_norm = torch.linalg.vector_norm(y_true) + _EPS
    rel_l2 = float(residual_norm / target_norm)

    return rmse, rel_l2


def save_comparison_grid(
    q: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str | Path,
) -> None:
    """Plot a 4-row grid of Q, ground truth, prediction and absolute error.

    Each argument is ``[N, H, W]``; one column per sample.
    """
    num_samples = q.shape[0]
    error = np.abs(y_pred - y_true)
    fields = (q, y_true, y_pred, error)
    row_labels = ("Heat Source Q", "Ground Truth T", "FNO Prediction", "Abs. Error")

    vmin_t = float(min(y_true.min(), y_pred.min()))
    vmax_t = float(max(y_true.max(), y_pred.max()))
    vmax_err = float(error.max())

    fig, axes = plt.subplots(4, num_samples, figsize=(3.5 * num_samples, 14))
    if num_samples == 1:
        axes = axes[:, np.newaxis]

    for row in range(4):
        for col in range(num_samples):
            ax = axes[row, col]
            field = fields[row][col]

            if row == 0:
                image = ax.imshow(field, cmap="inferno")
            elif row in (1, 2):
                image = ax.imshow(
                    field, cmap="viridis", vmin=vmin_t, vmax=vmax_t
                )
            else:
                image = ax.imshow(
                    field, cmap="magma", vmin=0.0, vmax=vmax_err
                )

            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticks([])
            ax.set_yticks([])

            if col == 0:
                ax.set_ylabel(row_labels[row])
            if row == 0:
                ax.set_title(f"Sample {col + 1}")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate the FNO-2D thermal surrogate."
    )
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--save-path", type=str, default="results.png")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args(argv)

    device = get_device()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    stats = stats_from_dict(checkpoint["stats"])
    hyperparameters = checkpoint.get("hyperparameters", {})

    include_initial = bool(hyperparameters.get("include_initial", False))
    in_channels = int(hyperparameters.get("in_channels", 1))
    out_channels = int(hyperparameters.get("out_channels", 1))
    modes1 = int(hyperparameters.get("modes1", 12))
    modes2 = int(hyperparameters.get("modes2", 12))
    width = int(hyperparameters.get("width", 32))

    test_path = Path(args.data_dir) / "test.npz"
    if not test_path.exists():
        raise FileNotFoundError(f"Test split not found: {test_path}")

    # Reuse the checkpoint statistics so the test split is normalized exactly
    # as during training.
    test_ds = ThermalDataset(
        test_path,
        include_initial=include_initial,
        normalize=True,
        norm="zscore",
        stats=stats,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = FNO2d(
        modes1=modes1,
        modes2=modes2,
        width=width,
        in_channels=in_channels,
        out_channels=out_channels,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    inputs: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)
            inputs.append(x.cpu())
            predictions.append(pred.cpu())
            targets.append(y.cpu())

    x_all = torch.cat(inputs, dim=0)
    y_pred_norm = torch.cat(predictions, dim=0)
    y_true_norm = torch.cat(targets, dim=0)

    # Convert back to physical units before reporting/plotting.
    q_phys = denormalize_input(x_all, stats)[:, 0]  # [N, H, W]
    y_pred = denormalize_target(y_pred_norm, stats)[:, 0]  # [N, H, W]
    y_true = denormalize_target(y_true_norm, stats)[:, 0]  # [N, H, W]

    rmse, rel_l2 = aggregate_metrics(y_pred, y_true)

    print(f"Test RMSE:          {rmse:.6f}")
    print(f"Test Relative L2:   {rel_l2:.6f}")

    num_samples = min(args.num_samples, y_true.shape[0])
    save_comparison_grid(
        q_phys[:num_samples].numpy(),
        y_true[:num_samples].numpy(),
        y_pred[:num_samples].numpy(),
        args.save_path,
    )
    print(f"Saved comparison grid to {args.save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())