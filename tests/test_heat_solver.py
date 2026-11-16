"""Unit tests for the 2D transient heat-equation solver."""

import numpy as np
import pytest

from thermal_surrogate_ml.physics.heat_solver import (
    BoundaryCondition,
    GaussianSource,
    HeatSolver,
    HeatSolverConfig,
    gaussian_source,
)


def _gaussian_bump(solver: HeatSolver) -> np.ndarray:
    """Return a smooth, localized initial temperature field."""
    return np.exp(
        -((solver.X - 0.5) ** 2 + (solver.Y - 0.5) ** 2) / (2.0 * 0.1**2)
    )


def test_stable_dt_satisfies_von_neumann() -> None:
    config = HeatSolverConfig(grid_size=64, length=1.0, alpha=1.0)
    solver = HeatSolver(config)

    assert solver.dt == pytest.approx(0.2 * solver.dx**2 / solver.alpha)
    assert solver.alpha * solver.dt / solver.dx**2 <= 0.25


def test_uniform_field_remains_static() -> None:
    config = HeatSolverConfig(
        grid_size=32,
        length=1.0,
        alpha=1.0,
        boundary=BoundaryCondition.DIRICHLET,
        boundary_value=1.0,
    )
    solver = HeatSolver(config)
    initial = np.full((config.grid_size, config.grid_size), 1.0)

    result = solver.solve(t_final=0.1, initial=initial, source=None)

    assert np.allclose(result.fields, 1.0)


def test_energy_conservation_neumann() -> None:
    config = HeatSolverConfig(
        grid_size=32,
        length=1.0,
        alpha=1.0,
        boundary=BoundaryCondition.NEUMANN,
    )
    solver = HeatSolver(config)

    result = solver.solve(t_final=0.1, initial=_gaussian_bump(solver), source=None)

    energies = result.fields.sum(axis=(1, 2)) * solver.dx**2
    assert np.allclose(energies, energies[0], rtol=1e-9, atol=1e-12)


def test_maximum_principle_no_overshoot() -> None:
    config = HeatSolverConfig(
        grid_size=32,
        length=1.0,
        alpha=1.0,
        boundary=BoundaryCondition.DIRICHLET,
        boundary_value=0.0,
    )
    solver = HeatSolver(config)
    initial = _gaussian_bump(solver)
    initial_max = float(initial.max())

    result = solver.solve(t_final=0.1, initial=initial, source=None)

    maxima = result.fields.max(axis=(1, 2))
    assert np.all(maxima <= initial_max + 1e-12)
    assert np.all(np.diff(maxima) <= 1e-12)


def test_source_does_not_exceed_physical_bound() -> None:
    config = HeatSolverConfig(
        grid_size=32,
        length=1.0,
        alpha=1.0,
        boundary=BoundaryCondition.DIRICHLET,
        boundary_value=0.0,
    )
    solver = HeatSolver(config)
    amplitude = 10.0
    source = gaussian_source(
        [GaussianSource(center_x=0.5, center_y=0.5, amplitude=amplitude, sigma=0.1)]
    )
    q_max = float(np.asarray(source(solver.X, solver.Y, 0.0)).max())
    initial = np.zeros((config.grid_size, config.grid_size))

    t_final = 0.05
    result = solver.solve(t_final=t_final, initial=initial, source=source)

    maxima = result.fields.max(axis=(1, 2))
    bound = result.times * q_max
    assert np.all(maxima <= bound + 1e-12)
