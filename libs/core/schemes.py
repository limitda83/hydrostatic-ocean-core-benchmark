#########################################################################
#  Module: schemes                                                      #
#  Description: Time integration for the 2D linear rotating shallow     #
#               water equations - forward-backward (explicit) and the   #
#               Casulli-type theta scheme (semi-implicit; theta=1 is    #
#               fully implicit). See docs/03_discretization_spec.md S3. #
#  Pipeline: grid/operators/solvers -> schemes -> driver                #
#########################################################################

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

import numpy as np

from libs.core.grid import CGrid
from libs.core.operators import (avg_u_to_v, avg_v_to_u, div_eta, gradx_u,
                                 grady_v)
from libs.core.solvers import build_solver

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class State:
    """Prognostic state on the C-grid."""

    u: np.ndarray
    v: np.ndarray
    eta: np.ndarray
    t: float = 0.0

    def astype(self, dtype) -> "State":
        return replace(self, u=self.u.astype(dtype), v=self.v.astype(dtype),
                       eta=self.eta.astype(dtype))


@dataclass(frozen=True)
class Physics:
    """Physical constants shared by every scheme."""

    g: float
    f: float
    H: float

    @property
    def c(self) -> float:
        """Surface gravity wave speed sqrt(gH)."""
        return float(np.sqrt(self.g * self.H))


@dataclass
class SchemeParams:
    """Time-integration parameters (axis A) and elliptic solver choice (axis B)."""

    name: str = "theta"          # "fb" | "theta" | "split_explicit" (3D only)
    theta: float = 0.5           # implicitness factor; 0.5=CN, 1.0=backward Euler
    theta_cor: float = 0.5       # Coriolis weighting inside the Picard loop
    n_picard: int = 2            # 1 => explicit Coriolis (1st order); 2 => 2nd order
    theta_v: float = 0.5         # implicitness of the vertical diffusion (3D)
    advection: str = "none"      # "none" | "centered2" | "upwind1" | "up3" | "up3_tvd" (S9.2, S11.3)
    n_split: int = 0             # barotropic substeps; 0 selects them from the CFL
    barotropic_average: str = "none"   # "none" | "mean" (spec S8.5)
    solver: str = "fft"
    rtol: float = 1e-12
    max_iter: int = 2000

    def validate(self) -> None:
        if self.name not in ("fb", "theta", "split_explicit"):
            raise ValueError(f"unknown scheme '{self.name}' "
                             f"(expected fb|theta|split_explicit)")
        if self.name == "theta" and not (0.0 < self.theta <= 1.0):
            # theta -> 0 degenerates to forward Euler, which is unconditionally
            # unstable for gravity waves (spec S3.2).
            raise ValueError(f"theta must lie in (0, 1]; got {self.theta}")
        if self.n_picard < 1:
            raise ValueError("n_picard must be >= 1")
        if self.advection not in ("none", "centered2", "upwind1", "up3", "up3_tvd"):
            raise ValueError(f"unknown advection '{self.advection}' "
                             f"(expected none|centered2|upwind1)")


def explicit_dt_max(grid: CGrid, physics: Physics) -> float:
    """Forward-backward stability limit (spec S3.2).

    dt <= 2 / (c * sqrt(4/dx^2 + 4/dy^2)), plus the Coriolis limit f*dt < 2.
    Used as the common yardstick so that cfl_factor is comparable across
    schemes, including the unconditionally stable ones.
    """
    s_max = np.sqrt(4.0 / grid.dx**2 + 4.0 / grid.dy**2)
    dt_gravity = 2.0 / (physics.c * s_max)
    dt_coriolis = 2.0 / abs(physics.f) if physics.f != 0.0 else np.inf
    return float(min(dt_gravity, dt_coriolis))


class Stepper:
    """One-step time integrator for a fixed (grid, physics, dt, params)."""

    def __init__(self, grid: CGrid, physics: Physics, params: SchemeParams,
                 dt: float) -> None:
        params.validate()
        self.grid = grid
        self.physics = physics
        self.params = params
        self.dt = float(dt)
        self.solver_iterations = 0
        self.solver_failures = 0

        if params.name == "theta":
            coef = physics.g * physics.H * params.theta**2 * self.dt**2
            self.solver = build_solver(params.solver, grid, coef,
                                       params.rtol, params.max_iter)
        else:
            self.solver = None

    # ------------------------------------------------------------------ core
    def _coriolis_terms(self, state_prev: State, u_it: np.ndarray,
                        v_it: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """theta_cor-weighted Coriolis partners (spec S3.1)."""
        tc = self.params.theta_cor
        v_cor = (1.0 - tc) * avg_v_to_u(state_prev.v) + tc * avg_v_to_u(v_it)
        u_cor = (1.0 - tc) * avg_u_to_v(state_prev.u) + tc * avg_u_to_v(u_it)
        return u_cor, v_cor

    def _step_fb(self, s: State) -> State:
        """Forward-backward explicit scheme (spec S3.2)."""
        g, f, H, dt = self.physics.g, self.physics.f, self.physics.H, self.dt
        gx_eta = gradx_u(s.eta, self.grid)
        gy_eta = grady_v(s.eta, self.grid)

        u_it, v_it = s.u, s.v
        for _ in range(self.params.n_picard):
            u_cor, v_cor = self._coriolis_terms(s, u_it, v_it)
            u_it = s.u + dt * (f * v_cor - g * gx_eta)
            v_it = s.v + dt * (-f * u_cor - g * gy_eta)

        eta_new = s.eta - dt * H * div_eta(u_it, v_it, self.grid)
        return State(u=u_it, v=v_it, eta=eta_new, t=s.t + dt)

    def _step_theta(self, s: State) -> State:
        """Casulli-type semi-implicit theta scheme (spec S3.3)."""
        g, f, H, dt = self.physics.g, self.physics.f, self.physics.H, self.dt
        th = self.params.theta
        gx_eta = gradx_u(s.eta, self.grid)
        gy_eta = grady_v(s.eta, self.grid)
        div_old = div_eta(s.u, s.v, self.grid)

        u_it, v_it, eta_new = s.u, s.v, s.eta
        for _ in range(self.params.n_picard):
            u_cor, v_cor = self._coriolis_terms(s, u_it, v_it)
            # Explicit predictors Gu, Gv - term order fixed by spec S6.1.
            gu = s.u + dt * f * v_cor - g * dt * (1.0 - th) * gx_eta
            gv = s.v - dt * f * u_cor - g * dt * (1.0 - th) * gy_eta

            rhs = s.eta - dt * H * ((1.0 - th) * div_old
                                    + th * div_eta(gu, gv, self.grid))
            eta_new, report = self.solver.solve(rhs)
            self.solver_iterations += report.iterations
            if not report.converged:
                self.solver_failures += 1

            u_it = gu - g * dt * th * gradx_u(eta_new, self.grid)
            v_it = gv - g * dt * th * grady_v(eta_new, self.grid)

        return State(u=u_it, v=v_it, eta=eta_new, t=s.t + dt)

    # ----------------------------------------------------------------- public
    def step(self, state: State) -> State:
        if self.params.name == "fb":
            return self._step_fb(state)
        return self._step_theta(state)

    def integrate(self, state: State, n_steps: int) -> State:
        for _ in range(n_steps):
            state = self.step(state)
        return state


def scheme_params_from_config(config) -> SchemeParams:
    """Build SchemeParams from config/schemes.toml."""
    params = SchemeParams(
        name=str(config.get("scheme.name")),
        theta=float(config.get("scheme.theta")),
        theta_cor=float(config.get("scheme.theta_cor")),
        theta_v=float(config.get("scheme.theta_v", 0.5)),
        advection=str(config.get("scheme.advection", "none")),
        n_split=int(config.get("scheme.n_split", 0)),
        barotropic_average=str(config.get("scheme.barotropic_average", "none")),
        n_picard=int(config.get("scheme.n_picard")),
        solver=str(config.get("solver.kind")),
        rtol=float(config.get("solver.rtol")),
        max_iter=int(config.get("solver.max_iter")),
    )
    params.validate()
    return params


def physics_from_config(config) -> Physics:
    return Physics(g=float(config.get("physics.g")),
                   f=float(config.get("physics.f0")),
                   H=float(config.get("physics.H")))
