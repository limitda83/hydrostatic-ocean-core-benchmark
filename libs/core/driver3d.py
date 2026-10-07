#########################################################################
#  Module: driver3d                                                     #
#  Description: Run one 3D configuration and collect its diagnostics,    #
#               mirroring libs/core/driver logic in main.py for the 2D  #
#               model. Keeps main.py thin.                              #
#  Pipeline: config -> driver3d -> model3d/cases3d -> metrics           #
#########################################################################

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from libs.core.cases3d import build_case3d
from libs.core.grid import build_grid
from libs.core.limits import memory_model, time_step_limits
from libs.core.model3d import Physics3D, State3D, Stepper3D
from libs.core.schemes import explicit_dt_max, scheme_params_from_config
from libs.utils.config import Config
from libs.utils.timing import timed_repeat

LOGGER = logging.getLogger(__name__)


def physics3d_from_config(config: Config) -> Physics3D:
    return Physics3D(
        g=float(config.get("physics.g")),
        f=float(config.get("physics.f0")),
        H=float(config.get("physics.H")),
        N2=float(config.get("physics3d.N2", 0.0)),
        nu=float(config.get("physics3d.nu", 0.0)),
        kappa=float(config.get("physics3d.kappa", 0.0)),
        rho0=float(config.get("physics.rho0", 1025.0)),
        tau_x=float(config.get("physics3d.tau_x", 0.0)),
        tau_y=float(config.get("physics3d.tau_y", 0.0)),
        bottom_drag=float(config.get("physics3d.bottom_drag", 0.0)),
        tracers=str(config.get("physics3d_v04.tracers", "buoyancy")),
        alpha_T=float(config.get("physics3d_v04.alpha_T", 2.0e-4)),
        beta_S=float(config.get("physics3d_v04.beta_S", 7.4e-4)),
        T0=float(config.get("physics3d_v04.T0", 10.0)),
        S0=float(config.get("physics3d_v04.S0", 35.0)),
        cp=float(config.get("physics3d_v04.cp", 3990.0)),
        A_h=float(config.get("physics3d_v04.A_h", 0.0)),
        K_h=float(config.get("physics3d_v04.K_h", 0.0)),
        q_heat=float(config.get("physics3d_v04.q_heat", 0.0)),
        q_salt=float(config.get("physics3d_v04.q_salt", 0.0)),
    )


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.sqrt(np.sum(b * b)))
    num = float(np.sqrt(np.sum((a - b) ** 2)))
    return num / den if den > 0.0 else num


def simulate3d(config: Config, nx: int, nz: int, cfl: float | None,
               case_name: str, n_steps_override: int | None = None,
               n_repeat: int = 1) -> dict[str, Any]:
    """Integrate one 3D configuration and return state, exact solution, metrics."""
    grid = build_grid(config, nx=nx, ny=nx, nz=nz)
    physics = physics3d_from_config(config)
    params = scheme_params_from_config(config)
    case, t_final = build_case3d(case_name, grid, physics, config)

    dt_max = explicit_dt_max(grid, physics)
    pinned = float(config.get("time.dt", 0.0))
    factor = cfl if cfl is not None else float(config.get("time.cfl_factor"))
    dt_req = pinned if pinned > 0.0 else factor * dt_max

    if n_steps_override is not None:
        n_steps, dt = n_steps_override, dt_req
        t_final = dt * n_steps
    else:
        n_steps = max(1, int(round(t_final / dt_req)))
        dt = t_final / n_steps

    stepper = Stepper3D(grid, physics, params, dt)
    initial = case.initial()
    final, timing = timed_repeat(lambda: stepper.integrate(initial, n_steps),
                                 n_repeat=n_repeat, n_warmup=0)
    exact = case.exact(t_final)

    cells = grid.nx * grid.ny * grid.nz
    solves = max(1, n_steps * params.n_picard)
    lim = time_step_limits(grid, physics,
                           u_scale=float(np.max(np.abs(initial.u))) or 1.0)
    dt_allowed = lim.for_scheme(params.name, params.advection, physics.A_h)
    mem = memory_model(grid, params.name)
    # Simulated time per unit wall-clock time at the largest stable step.
    # This is the metric that decides what you can actually run; cost per step
    # alone hides the factor of 10 difference in allowed dt between schemes.
    per_step = timing.median_s / n_steps
    sdpd = dt_allowed / per_step if per_step > 0 else float("nan")
    metrics: dict[str, Any] = {
        "case": case_name, "nx": grid.nx, "ny": grid.ny, "nz": grid.nz,
        "cells": cells, "dx": grid.dx, "dz": grid.dz,
        "dt": dt, "dt_max_explicit": dt_max, "cfl": dt / dt_max,
        "n_steps": n_steps, "t_final": t_final,
        "scheme": params.name, "theta": params.theta, "theta_v": params.theta_v,
        "n_picard": params.n_picard, "solver": params.solver,
        "h_eff": stepper.h_eff, "n_split": stepper.n_split,
        "barotropic_substeps": stepper.barotropic_substeps,
        "solver_iterations": stepper.solver_iterations,
        "pcg_per_solve": stepper.solver_iterations / solves,
        "tridiagonal_solves": stepper.tridiagonal_solves,
        "l2_rel_u": _rel_l2(final.u, exact.u),
        "l2_rel_T": (_rel_l2(final.T - physics.T0, exact.T - physics.T0)
                     if final.T is not None else float("nan")),
        "l2_rel_b": _rel_l2(final.b, exact.b) if np.any(exact.b) else float("nan"),
        "l2_rel_eta": _rel_l2(final.eta, exact.eta) if np.any(exact.eta) else float("nan"),
        "max_abs_eta": float(np.max(np.abs(final.eta))),
        "wall_s": timing.median_s, "wall_mad_s": timing.mad_s,
        "us_per_step": 1e6 * timing.median_s / n_steps,
        "ns_per_cell_step": 1e9 * timing.median_s / (n_steps * cells),
        "timing_reportable": timing.reportable,
        "dt_limit_scheme": dt_allowed,
        "dt_limits": lim.to_dict(),
        "sim_seconds_per_wall_second": sdpd,
        "sim_days_per_wall_day": sdpd,
        "bytes": mem["bytes"], "bytes_per_cell": mem["bytes_per_cell"],
        "gib": mem["gib"], "max_nx_32gib_nz30": mem["max_nx_32gib_nz30"],
    }
    return {"grid": grid, "physics": physics, "params": params, "case": case,
            "state": final, "exact": exact, "metrics": metrics}
