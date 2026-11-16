"""2D transient heat-equation solver (explicit finite differences).

Solves ``dT/dt = alpha * laplacian(T) + Q(x, y, t)`` on ``[0, L] x [0, L]``.

Physics arrays are ``[ny, nx]`` NumPy arrays in ``meshgrid(..., indexing="xy")``
layout, so ``X[i, j] == x[j]`` and ``Y[i, j] == y[i]``. The surrogate consumes
the same field as a single-channel ``[1, 1, ny, nx]`` tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

FloatField = NDArray[np.float64]
SourceCallable = Callable[[FloatField, FloatField, float], FloatField]
Source = FloatField | SourceCallable | None


class BoundaryCondition(str, Enum):
    """Boundary condition applied uniformly to all four domain edges."""

    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"


@dataclass(frozen=True)
class GaussianSource:
    """A steady Gaussian heat source (e.g. an electronic chip on the substrate)."""

    center_x: float
    center_y: float
    amplitude: float
    sigma: float


def gaussian_source(spots: Sequence[GaussianSource]) -> SourceCallable:
    """Build a ``Q(x, y, t)`` callable as the superposition of Gaussian spots.

    The time argument is accepted for API compatibility and ignored by the
    steady profile.
    """

    def source(
        X: FloatField,
        Y: FloatField,
        _t: float,
    ) -> FloatField:
        field = np.zeros_like(X)
        for spot in spots:
            r2 = (X - spot.center_x) ** 2 + (Y - spot.center_y) ** 2
            field += spot.amplitude * np.exp(-r2 / (2.0 * spot.sigma**2))
        return field

    return source


@dataclass(frozen=True)
class HeatSolverConfig:
    """Configuration for :class:`HeatSolver`."""

    grid_size: int = 64
    length: float = 1.0
    alpha: float = 1.0
    dt: float | None = None
    cfl_factor: float = 0.2
    boundary: BoundaryCondition = BoundaryCondition.DIRICHLET
    boundary_value: float = 0.0

    def __post_init__(self) -> None:
        if self.grid_size < 3:
            raise ValueError("grid_size must be >= 3")
        if self.length <= 0.0:
            raise ValueError("length must be > 0")
        if self.alpha <= 0.0:
            raise ValueError("alpha must be > 0")
        if not 0.0 < self.cfl_factor <= 0.25:
            raise ValueError("cfl_factor must lie in (0, 0.25] for 2D stability")
        if self.dt is not None and self.dt <= 0.0:
            raise ValueError("dt must be > 0 when provided")


@dataclass(frozen=True)
class SimulationResult:
    """Time-series output of :meth:`HeatSolver.solve`."""

    fields: FloatField  # shape [n_snapshots, ny, nx]
    times: NDArray[np.float64]  # shape [n_snapshots]
    dt: float


class HeatSolver:
    """Explicit finite-difference integrator for the 2D heat equation.

    The time step follows the von Neumann limit ``dt = cfl_factor * dx**2 /
    alpha``, with ``cfl_factor = 0.2`` safely below the 0.25 stability bound.
    """

    def __init__(self, config: HeatSolverConfig) -> None:
        self._config = config

        self._x = np.linspace(0.0, config.length, config.grid_size)
        self._y = np.linspace(0.0, config.length, config.grid_size)
        self._X, self._Y = np.meshgrid(self._x, self._y, indexing="xy")

        dx = self._x[1] - self._x[0]
        if config.dt is not None:
            dt = config.dt
        else:
            dt = config.cfl_factor * dx**2 / config.alpha

        r = config.alpha * dt / dx**2
        if r > 0.25 + 1e-15:
            raise ValueError(
                f"Unstable time step: alpha*dt/dx^2 = {r:.6f} exceeds the "
                "explicit 2D von Neumann limit of 0.25."
            )

        self._dx = float(dx)
        self._dt = float(dt)

    # -- geometry / physics -------------------------------------------------

    @property
    def config(self) -> HeatSolverConfig:
        return self._config

    @property
    def grid_size(self) -> int:
        return self._config.grid_size

    @property
    def length(self) -> float:
        return self._config.length

    @property
    def alpha(self) -> float:
        return self._config.alpha

    @property
    def dx(self) -> float:
        return self._dx

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def x(self) -> NDArray[np.float64]:
        return self._x

    @property
    def y(self) -> NDArray[np.float64]:
        return self._y

    @property
    def X(self) -> FloatField:
        return self._X

    @property
    def Y(self) -> FloatField:
        return self._Y

    # -- Laplacian stencils -------------------------------------------------

    def laplacian_dirichlet(self, field: FloatField) -> FloatField:
        """5-point Laplacian on interior nodes; boundary nodes stay zero.

        The caller applies the Dirichlet values after the update.
        """
        lap = np.zeros_like(field)
        lap[1:-1, 1:-1] = (
            field[2:, 1:-1] + field[:-2, 1:-1]
            + field[1:-1, 2:] + field[1:-1, :-2]
            - 4.0 * field[1:-1, 1:-1]
        ) / self._dx**2
        return lap

    def laplacian_neumann(self, field: FloatField) -> FloatField:
        """5-point Laplacian over the full field with zero-flux edges.

        ``np.pad(..., mode="edge")`` mirrors the boundary node into a ghost
        cell, which is exactly the Neumann condition and keeps the scheme
        conservative.
        """
        padded = np.pad(field, pad_width=1, mode="edge")
        return (
            padded[2:, 1:-1] + padded[:-2, 1:-1]
            + padded[1:-1, 2:] + padded[1:-1, :-2]
            - 4.0 * padded[1:-1, 1:-1]
        ) / self._dx**2

    # -- stepping -----------------------------------------------------------

    def _evaluate_source(self, source: Source, t: float) -> FloatField:
        if source is None:
            return np.zeros((self._config.grid_size, self._config.grid_size))
        if callable(source):
            return np.asarray(source(self._X, self._Y, t), dtype=np.float64)
        return np.asarray(source, dtype=np.float64)

    def step(
        self,
        field: FloatField,
        source: Source = None,
        t: float = 0.0,
    ) -> FloatField:
        """Advance one explicit Euler step and enforce the boundary condition.

        ``source`` may be an array, a ``(X, Y, t) -> field`` callable, or
        ``None`` for ``Q = 0``; ``t`` is the time at which it is evaluated.
        """
        source_field = self._evaluate_source(source, t)

        if self._config.boundary == BoundaryCondition.DIRICHLET:
            lap = self.laplacian_dirichlet(field)
            updated = (
                field
                + self._config.alpha * self._dt * lap
                + self._dt * source_field
            )
            updated[0, :] = self._config.boundary_value
            updated[-1, :] = self._config.boundary_value
            updated[:, 0] = self._config.boundary_value
            updated[:, -1] = self._config.boundary_value
            return updated

        lap = self.laplacian_neumann(field)
        return (
            field
            + self._config.alpha * self._dt * lap
            + self._dt * source_field
        )

    # -- full simulation ----------------------------------------------------

    def solve(
        self,
        t_final: float,
        initial: FloatField,
        source: Source = None,
    ) -> SimulationResult:
        """Integrate from ``t = 0`` to ``t_final``, returning the full history.

        Index 0 of ``fields`` is the initial condition.
        """
        if t_final < 0.0:
            raise ValueError("t_final must be non-negative")

        field = np.asarray(initial, dtype=np.float64)
        expected = (self._config.grid_size, self._config.grid_size)
        if field.shape != expected:
            raise ValueError(
                f"initial must have shape {expected}, got {field.shape}"
            )
        field = field.copy()

        num_steps = int(round(t_final / self._dt))
        fields = np.empty((num_steps + 1,) + expected, dtype=np.float64)
        fields[0] = field

        for n in range(num_steps):
            field = self.step(field, source=source, t=(n + 1) * self._dt)
            fields[n + 1] = field

        times = np.arange(num_steps + 1, dtype=np.float64) * self._dt
        return SimulationResult(fields=fields, times=times, dt=self._dt)

    def solve_final(
        self,
        t_final: float,
        initial: FloatField,
        source: Source = None,
    ) -> FloatField:
        """Integrate to ``t_final`` and return only the final field.

        Avoids materializing the time history, keeping memory bounded during
        dataset generation.
        """
        if t_final < 0.0:
            raise ValueError("t_final must be non-negative")

        field = np.asarray(initial, dtype=np.float64)
        expected = (self._config.grid_size, self._config.grid_size)
        if field.shape != expected:
            raise ValueError(
                f"initial must have shape {expected}, got {field.shape}"
            )
        field = field.copy()

        num_steps = int(round(t_final / self._dt))
        for n in range(num_steps):
            field = self.step(field, source=source, t=(n + 1) * self._dt)
        return field

    def solve_steady_state(
        self,
        initial: FloatField,
        source: Source = None,
        tol: float = 1e-8,
        max_steps: int | None = None,
    ) -> FloatField:
        """Integrate until ``max|T_new - T_old| < tol``.

        The default iteration cap scales as ``O((grid_size - 1)**2)`` so the
        method terminates even when ``tol`` is unreachable.
        """
        field = np.asarray(initial, dtype=np.float64)
        expected = (self._config.grid_size, self._config.grid_size)
        if field.shape != expected:
            raise ValueError(
                f"initial must have shape {expected}, got {field.shape}"
            )
        field = field.copy()

        if max_steps is None:
            max_steps = int(20 * (self._config.grid_size - 1) ** 2)
        if max_steps < 1:
            raise ValueError("max_steps must be a positive integer")

        for _ in range(max_steps):
            updated = self.step(field, source=source, t=0.0)
            if float(np.max(np.abs(updated - field))) < tol:
                return updated
            field = updated
        return field
