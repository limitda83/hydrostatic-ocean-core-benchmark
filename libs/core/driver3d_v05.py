#########################################################################
#  Module: driver3d_v05                                                 #
#  Description: Run one spec v0.5 case end to end and report both the   #
#               case metric and the cost (docs/03 S10.8). Every run is  #
#               self-describing: the returned record carries the        #
#               domain, scheme and solver it was produced with (R6).    #
#  Pipeline: config -> driver3d_v05 -> verify3d5 / bench3d5             #
#########################################################################

from __future__ import annotations

import time

import numpy as np

from libs.core.bathymetry import bathymetry_from_config
from libs.core.cases3d_v05 import build_case3d_v05
from libs.core.domain import Domain
from libs.core.grid import CGrid
from libs.core.model3d import Physics3D
from libs.core.model3d_v05 import PhysicsV05, Stepper3DV05
from libs.core.schemes import scheme_params_from_config


def build_domain3d(config, nx: int, nz: int, *, vcoord: str | None = None,
                   bathymetry: str | None = None, bc: str | None = None,
                   **bath_kw) -> Domain:
    """Assemble the v0.5 domain, allowing per-run overrides of the axes."""
    grid = CGrid(nx=nx, ny=nx, Lx=float(config.get("grid.Lx")),
                 Ly=float(config.get("grid.Ly")), nz=nz,
                 depth=float(config.get("physics.H")))
    if bathymetry is None and not bath_kw:
        H, mask = bathymetry_from_config(grid, config)
    else:
        from libs.core.bathymetry import build_bathymetry
        kind = bathymetry or str(config.get("bathymetry.kind"))
        defaults = dict(
            h_rel=float(config.get("bathymetry.h_rel", 0.9)),
            length=float(config.get("bathymetry.length", 0.0)),
            slope=float(config.get("bathymetry.slope", 0.5)),
            spectrum_slope=float(config.get("bathymetry.spectrum_slope", 1.5)),
            r_target=float(config.get("bathymetry.r_target", 0.1)),
            seed=int(config.get("bathymetry.seed", 20260911)),
            island=bool(config.get("bathymetry.island", False)))
        defaults.update(bath_kw)
        H, mask = build_bathymetry(grid, kind, float(config.get("physics.H")),
                                   **defaults)
    bx = bc or str(config.get("domain.bc_x"))
    by = bc or str(config.get("domain.bc_y"))
    return Domain(grid=grid, H=H, mask=mask,
                  vcoord=vcoord or str(config.get("domain.vcoord")),
                  bc_x=bx, bc_y=by,
                  min_partial=float(config.get("domain.min_partial", 0.1)),
                  face_rule=str(config.get("domain.face_rule", "min")))


def physics_v05_from_config(config, **over) -> PhysicsV05:
    base = Physics3D(
        g=float(config.get("physics.g")), f=float(config.get("physics.f0")),
        H=float(config.get("physics.H")), N2=float(config.get("physics3d.N2")),
        nu=float(config.get("physics3d.nu")),
        kappa=float(config.get("physics3d.kappa")),
        rho0=float(config.get("physics.rho0")),
        tau_x=float(config.get("physics3d.tau_x")),
        tau_y=float(config.get("physics3d.tau_y")),
        bottom_drag=float(config.get("physics3d.bottom_drag")),
        tracers=str(config.get("physics3d_v04.tracers")),
        alpha_T=float(config.get("physics3d_v04.alpha_T")),
        beta_S=float(config.get("physics3d_v04.beta_S")),
        T0=float(config.get("physics3d_v04.T0")),
        S0=float(config.get("physics3d_v04.S0")),
        cp=float(config.get("physics3d_v04.cp")),
        A_h=float(config.get("physics3d_v04.A_h")),
        K_h=float(config.get("physics3d_v04.K_h")),
        q_heat=float(config.get("physics3d_v04.q_heat")),
        q_salt=float(config.get("physics3d_v04.q_salt")))
    kw = dict(eos=str(config.get("physics3d_v05.eos")),
              pgf=str(config.get("physics3d_v05.pgf")),
              pgf_correction=bool(config.get("physics3d_v05.pgf_correction", True)),
              eta_ext=float(config.get("physics3d_v05.eta_ext")),
              u_ext=float(config.get("physics3d_v05.u_ext")),
              n_relax=int(config.get("physics3d_v05.n_relax")),
              tau_relax=float(config.get("physics3d_v05.tau_relax")))
    # spec v0.6 closure (S11.2); absent section keeps v0.5 behaviour
    try:
        v6 = config.section("physics3d_v06")
    except Exception:            # noqa: BLE001 - older config trees
        v6 = {}
    kw.update(closure=str(v6.get("closure", "none")),
              mxl=str(v6.get("mxl", "integral")),
              **{k: float(v6[k]) for k in ("c_k", "c_eps", "pr_t", "kappa_vk", "z_0", "e_min",
                                           "n2_min", "l_min", "e_bb") if k in v6})
    kw.update(over)
    return PhysicsV05(base=base, **kw)


def barotropic_dt_max(domain: Domain, g: float) -> float:
    """Surface gravity CFL of the deepest column."""
    c = np.sqrt(g * float(np.max(domain.H[domain.mask > 0])))
    return 1.0 / (c * np.sqrt(1.0 / domain.grid.dx**2 + 1.0 / domain.grid.dy**2))


def simulate3d_v05(config, case_name: str, domain: Domain, physics: PhysicsV05,
                   cfl: float, *, n_steps_override: int | None = None,
                   n_repeat: int = 1, n_warmup: int = 0) -> dict:
    """Integrate one case and return metric plus cost."""
    params = scheme_params_from_config(config)
    dt = cfl * barotropic_dt_max(domain, physics.g)
    stepper = Stepper3DV05(domain, physics, params, dt)
    case = build_case3d_v05(case_name, domain, physics, config)
    n_steps = (n_steps_override if n_steps_override is not None
               else max(1, int(round(case.t_final / dt))))

    s0 = case.initial()
    for _ in range(n_warmup):
        stepper.integrate(s0, min(n_steps, 3))
    walls = []
    final = s0
    for _ in range(max(1, n_repeat)):
        stepper.solver_iterations = 0
        stepper.barotropic_substeps = 0
        stepper.eos_cells = 0
        t0 = time.perf_counter()
        final = stepper.integrate(s0, n_steps)
        walls.append(time.perf_counter() - t0)
    wall = float(np.median(walls))

    result = case.evaluate(final)
    d = domain.summary()
    return {
        "case": case_name, "metric_name": result.metric_name,
        "metric": result.metric, "extra": result.extra,
        "nx": domain.grid.nx, "nz": domain.grid.nz,
        "cells": domain.grid.nx * domain.grid.ny * domain.grid.nz,
        "n_steps": n_steps, "dt": dt, "cfl": cfl,
        "scheme": params.name, "solver": params.solver,
        "eos": physics.eos, "pgf": physics.pgf,
        "pgf_correction": physics.pgf_correction,
        "wall_s": wall, "wall_mad_s": float(np.median(np.abs(np.array(walls) - wall))),
        "solver_iters_per_step": stepper.solver_iterations / max(1, n_steps),
        "barotropic_substeps_per_step": stepper.barotropic_substeps / max(1, n_steps),
        "tridiagonal_solves": stepper.tridiagonal_solves,
        "eos_flops": stepper.eos_flops,
        "moving_coordinate": stepper.moving_coordinate,
        **d,
    }
