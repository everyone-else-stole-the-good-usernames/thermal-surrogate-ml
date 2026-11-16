"""Benchmark the FNO surrogate against the explicit heat solver.

Measures inference speedup on 64x64 fields and verifies zero-shot
super-resolution on 128x128 fields. The solver stays NumPy/CPU while the FNO
runs on the selected PyTorch device, keeping the two pipelines decoupled.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from thermal_surrogate_ml.data.dataset import NormalizationStats
from thermal_surrogate_ml.models.fno import FNO2d
from thermal_surrogate_ml.physics.heat_solver import (
    BoundaryCondition,
    GaussianSource,
    HeatSolver,
    HeatSolverConfig,
    gaussian_source,
)
from thermal_surrogate_ml.training.train import get_device, stats_from_dict

FloatField = NDArray[np.float64]

_EPS = 1e-8
_WARMUP_RUNS = 3


def random_gaussian_source(
    solver: HeatSolver,
    rng: np.random.Generator,
) -> FloatField:
    """Sample one random Gaussian heat spot using the training-distribution ranges."""
    length = solver.length
    spot = GaussianSource(
        center_x=length * float(rng.uniform(0.1, 0.9)),
        center_y=length * float(rng.uniform(0.1, 0.9)),
        amplitude=float(rng.uniform(1.0, 10.0)),
        sigma=float(rng.uniform(0.05, 0.15)),
    )
    return np.asarray(
        gaussian_source([spot])(solver.X, solver.Y, 0.0), dtype=np.float64
    )


def build_solver(grid_size: int) -> HeatSolver:
    """Create a Dirichlet-bounded heat solver matching the training config."""
    config = HeatSolverConfig(
        grid_size=grid_size,
        length=1.0,
        alpha=1.0,
        boundary=BoundaryCondition.DIRICHLET,
        boundary_value=0.0,
    )
    return HeatSolver(config)


def time_solver(
    solver: HeatSolver,
    source: FloatField,
    runs: int,
) -> float:
    """Return the mean ``solve_steady_state`` wall time in seconds."""
    initial = np.zeros((solver.grid_size, solver.grid_size), dtype=np.float64)

    # Warm up so allocator/JIT effects don't pollute the first measurement.
    solver.solve_steady_state(initial=initial, source=source)

    start = time.perf_counter()
    for _ in range(runs):
        solver.solve_steady_state(initial=initial, source=source)
    return (time.perf_counter() - start) / runs


def normalize_source(
    source: FloatField,
    stats: NormalizationStats,
    device: torch.device,
) -> torch.Tensor:
    """Z-score normalize a source field into the ``[1, 1, H, W]`` tensor form."""
    mean = torch.as_tensor(stats.input_mean, dtype=torch.float32, device=device)
    std = torch.as_tensor(stats.input_std, dtype=torch.float32, device=device)

    x = torch.as_tensor(source, dtype=torch.float32, device=device)
    x = (x - mean[0]) / (std[0] + _EPS)
    return x.unsqueeze(0).unsqueeze(0)


@torch.no_grad()
def time_fno(
    model: torch.nn.Module,
    x: torch.Tensor,
    runs: int,
    device: torch.device,
) -> float:
    """Return the mean FNO forward-pass wall time in seconds.

    CUDA kernels are async, so GPU latency is measured with events; CPU latency
    uses ``perf_counter``.
    """
    model.eval()
    for _ in range(_WARMUP_RUNS):
        _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(runs):
            _ = model(x)
        end_event.record()
        torch.cuda.synchronize(device)
        elapsed = float(start_event.elapsed_time(end_event)) / 1000.0
        return elapsed / runs

    start = time.perf_counter()
    for _ in range(runs):
        _ = model(x)
    return (time.perf_counter() - start) / runs


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for the FNO benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark FNO inference speedup and zero-shot super-resolution."
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best_model.pt"
    )
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if args.runs < 1:
        raise ValueError("--runs must be a positive integer")

    device = get_device()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Match the loading protocol used by evaluate.py.
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    stats = stats_from_dict(checkpoint["stats"])
    hyperparameters = checkpoint.get("hyperparameters", {})

    modes1 = int(hyperparameters.get("modes1", 12))
    modes2 = int(hyperparameters.get("modes2", 12))
    width = int(hyperparameters.get("width", 32))
    in_channels = int(hyperparameters.get("in_channels", 1))
    out_channels = int(hyperparameters.get("out_channels", 1))

    if in_channels != 1:
        raise ValueError(
            "Benchmark assumes a single input channel (Q only); "
            f"checkpoint has in_channels={in_channels}."
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

    rng = np.random.default_rng(args.seed)

    # -- 64x64 speed benchmark ----------------------------------------------
    solver_64 = build_solver(64)
    source_64 = random_gaussian_source(solver_64, rng)
    solver_64_s = time_solver(solver_64, source_64, args.runs)

    x_64 = normalize_source(source_64, stats, device)
    fno_64_s = time_fno(model, x_64, args.runs, device)

    # -- 128x128 zero-shot super-resolution ---------------------------------
    solver_128 = build_solver(128)  # meshgrid only; no numerical solve is timed.
    source_128 = random_gaussian_source(solver_128, rng)
    x_128 = normalize_source(source_128, stats, device)

    with torch.no_grad():
        prediction_128 = model(x_128)

    expected_shape = (1, out_channels, 128, 128)
    actual_shape = tuple(prediction_128.shape)
    if actual_shape != expected_shape:
        raise RuntimeError(
            "Zero-shot output shape mismatch: "
            f"expected {expected_shape}, got {actual_shape}."
        )

    fno_128_s = time_fno(model, x_128, args.runs, device)

    # -- report -------------------------------------------------------------
    solver_64_ms = solver_64_s * 1000.0
    fno_64_ms = fno_64_s * 1000.0
    fno_128_ms = fno_128_s * 1000.0

    speedup_64 = solver_64_ms / fno_64_ms
    speedup_128 = solver_64_ms / fno_128_ms

    rows: list[tuple[str, str, str, str]] = [
        ("64x64", "Numerical Solver (HeatSolver)", f"{solver_64_ms:.3f}", "1.00x"),
        ("64x64", "FNO (64x64-trained)", f"{fno_64_ms:.3f}", f"{speedup_64:.1f}x"),
        ("128x128", "FNO (zero-shot)", f"{fno_128_ms:.3f}", f"{speedup_128:.1f}x"),
    ]

    print()
    print("| Grid Resolution | Method                        | Latency (ms) | Speedup Factor |")
    print("|----------------:|-------------------------------|-------------:|---------------:|")
    for resolution, method, latency, speedup in rows:
        print(
            f"| {resolution:<14} | {method:<29} | {latency:>11} | {speedup:>13} |"
        )
    print()
    print("Speedup Factor is relative to the 64x64 numerical solver baseline.")
    print(
        "Zero-shot super-resolution shape check passed: "
        f"output shape {actual_shape}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())