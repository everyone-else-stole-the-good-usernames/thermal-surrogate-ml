"""Parametric dataset generation for the 2D thermal surrogate.

Synthesizes ``(source Q, target T)`` pairs with the NumPy :class:`HeatSolver`,
deliberately free of PyTorch so the physics and ML pipelines stay decoupled.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, Sequence, cast

import numpy as np
from numpy.typing import NDArray

from thermal_surrogate_ml.physics.heat_solver import (
    BoundaryCondition,
    GaussianSource,
    HeatSolver,
    HeatSolverConfig,
    gaussian_source,
)

FloatField = NDArray[np.float64]


def random_scenario(
    rng: np.random.Generator,
    length: float,
    num_spots_range: tuple[int, int] = (1, 4),
    location_range: tuple[float, float] = (0.1, 0.9),
    amplitude_range: tuple[float, float] = (1.0, 10.0),
    sigma_range: tuple[float, float] = (0.05, 0.15),
) -> list[GaussianSource]:
    """Sample 1-4 random Gaussian heat spots inside the substrate."""

    low, high = num_spots_range
    n_spots = int(rng.integers(low, high + 1))

    spots: list[GaussianSource] = []
    for _ in range(n_spots):
        spots.append(
            GaussianSource(
                center_x=length * float(rng.uniform(*location_range)),
                center_y=length * float(rng.uniform(*location_range)),
                amplitude=float(rng.uniform(*amplitude_range)),
                sigma=float(rng.uniform(*sigma_range)),
            )
        )
    return spots


def _source_field(
    solver: HeatSolver,
    spots: Sequence[GaussianSource],
) -> FloatField:
    """Evaluate the steady source ``Q(x, y)`` on the solver meshgrid."""

    return np.asarray(
        gaussian_source(spots)(solver.X, solver.Y, 0.0), dtype=np.float64
    )


def _random_initial(
    rng: np.random.Generator,
    solver: HeatSolver,
    length: float,
) -> FloatField:
    """Sample a smooth, localized initial temperature field."""

    center_x = length * float(rng.uniform(0.2, 0.8))
    center_y = length * float(rng.uniform(0.2, 0.8))
    sigma = length * float(rng.uniform(0.1, 0.3))
    amplitude = float(rng.uniform(0.1, 1.0))
    return amplitude * np.exp(
        -((solver.X - center_x) ** 2 + (solver.Y - center_y) ** 2)
        / (2.0 * sigma**2)
    )


def generate_dataset(
    out_dir: str | Path = "data",
    n_train: int = 500,
    n_val: int = 100,
    n_test: int = 100,
    grid_size: int = 64,
    length: float = 1.0,
    alpha: float = 1.0,
    num_spots_range: tuple[int, int] = (1, 4),
    location_range: tuple[float, float] = (0.1, 0.9),
    amplitude_range: tuple[float, float] = (1.0, 10.0),
    sigma_range: tuple[float, float] = (0.05, 0.15),
    mode: Literal["transient", "steady"] = "transient",
    t_final: float = 0.1,
    steady_tol: float = 1e-8,
    randomize_initial: bool = False,
    seed: int = 0,
) -> dict[str, Path]:
    """Generate and save parametric thermal scenarios as compressed ``.npz``.

    Returns ``{split: path}``. Each split uses an independent RNG (``seed``,
    ``seed + 1``, ``seed + 2``) to prevent leakage between train/val/test.
    """

    if mode not in ("transient", "steady"):
        raise ValueError(f"mode must be 'transient' or 'steady', got {mode!r}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    config = HeatSolverConfig(
        grid_size=grid_size,
        length=length,
        alpha=alpha,
        boundary=BoundaryCondition.DIRICHLET,
        boundary_value=0.0,
    )
    solver = HeatSolver(config)

    counts = {"train": n_train, "val": n_val, "test": n_test}
    seeds = {"train": seed, "val": seed + 1, "test": seed + 2}
    paths: dict[str, Path] = {}

    for split, count in counts.items():
        rng = np.random.default_rng(seeds[split])

        sources = np.empty((count, grid_size, grid_size), dtype=np.float32)
        initials = np.empty((count, grid_size, grid_size), dtype=np.float32)
        targets = np.empty((count, grid_size, grid_size), dtype=np.float32)

        for i in range(count):
            spots = random_scenario(
                rng,
                length,
                num_spots_range=num_spots_range,
                location_range=location_range,
                amplitude_range=amplitude_range,
                sigma_range=sigma_range,
            )
            q = _source_field(solver, spots)

            if randomize_initial:
                t0 = _random_initial(rng, solver, length)
            else:
                t0 = np.zeros((grid_size, grid_size), dtype=np.float64)

            if mode == "steady":
                target = solver.solve_steady_state(
                    initial=t0, source=q, tol=steady_tol
                )
            else:
                target = solver.solve_final(t_final=t_final, initial=t0, source=q)

            sources[i] = q.astype(np.float32)
            initials[i] = t0.astype(np.float32)
            targets[i] = target.astype(np.float32)

        path = out / f"{split}.npz"
        np.savez_compressed(
            path,
            source=sources,
            initial=initials,
            target=targets,
        )
        paths[split] = path

    return paths


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for dataset generation."""

    parser = argparse.ArgumentParser(
        description="Generate parametric thermal surrogate datasets."
    )
    parser.add_argument("--n-train", type=int, default=500)
    parser.add_argument("--n-val", type=int, default=100)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--length", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--mode", choices=["transient", "steady"], default="transient")
    parser.add_argument("--t-final", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="data")
    args = parser.parse_args(argv)

    paths = generate_dataset(
        out_dir=args.out_dir,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        grid_size=args.grid_size,
        length=args.length,
        alpha=args.alpha,
        mode=cast(Literal["transient", "steady"], args.mode),
        t_final=args.t_final,
        seed=args.seed,
    )

    for split, path in paths.items():
        print(f"{split}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())