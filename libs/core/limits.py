#########################################################################
#  Module: limits                                                       #
#  Description: The stability constraints that decide how large a time  #
#               step each scheme may take at a given resolution, and    #
#               the memory a configuration needs per cell.              #
#                                                                       #
#               Cost per step alone does not decide time-to-solution:   #
#               a scheme that costs twice as much per step but is       #
#               allowed a ten times larger step wins. This module makes #
#               that trade explicit so the benchmark can report         #
#               simulated time per wall-clock time rather than          #
#               milliseconds per step.                                  #
#  Pipeline: grid/physics -> limits -> driver3d, tools/design_surface   #
#########################################################################

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from libs.core.grid import CGrid
from libs.core.model3d import Physics3D


@dataclass(frozen=True)
class TimeStepLimits:
    """Every explicit stability limit, in seconds, plus what each scheme keeps."""

    surface_gravity: float     # dt < 2 / (c * sqrt(4/dx^2 + 4/dy^2))
    internal_wave: float       # same with the mode-1 internal speed c_n = N H / pi
    advective: float           # dx / |u|max  (only when advection is on)
    horizontal_diffusion: float  # dx^2 / (4 A_h)  - explicit
    vertical_diffusion: float    # dz^2 / (2 nu)   - REMOVED by the implicit solve
    coriolis: float            # 2 / f

    def for_scheme(self, name: str, advection: str, A_h: float) -> float:
        """Largest stable dt for a scheme, ignoring accuracy."""
        caps = [self.coriolis]
        if name == "fb":
            caps.append(self.surface_gravity)
        else:
            # theta and split_explicit both step over the surface gravity wave:
            # theta by solving it implicitly, split_explicit by substepping it.
            caps.append(self.internal_wave)
        if advection != "none":
            caps.append(self.advective)
        if A_h > 0.0:
            caps.append(self.horizontal_diffusion)
        return float(min(caps))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def time_step_limits(grid: CGrid, physics: Physics3D,
                     u_scale: float = 1.0) -> TimeStepLimits:
    """All explicit limits at this resolution (spec S3.2, S7.4, S9.3)."""
    s_max = np.sqrt(4.0 / grid.dx**2 + 4.0 / grid.dy**2)
    c_ext = np.sqrt(physics.g * physics.H)
    # Mode-1 internal wave speed for constant N: c_n = N H / (n pi).
    c_int = (np.sqrt(physics.N2) * physics.H / np.pi) if physics.N2 > 0.0 else c_ext
    inf = float("inf")
    return TimeStepLimits(
        surface_gravity=float(2.0 / (c_ext * s_max)),
        internal_wave=float(2.0 / (c_int * s_max)) if c_int > 0 else inf,
        advective=float(grid.dx / u_scale) if u_scale > 0 else inf,
        horizontal_diffusion=(float(grid.dx**2 / (4.0 * physics.A_h))
                              if physics.A_h > 0 else inf),
        vertical_diffusion=(float(grid.dz**2 / (2.0 * physics.nu))
                            if physics.nu > 0 else inf),
        coriolis=float(2.0 / abs(physics.f)) if physics.f else inf,
    )


# Persistent working arrays the 3D stepper holds, counted in units of a full
# [nz, ny, nx] field. Measured from libs/core/model3d.py; the compiled backends
# declare the same set, so bytes/cell is comparable across them.
FIELDS_3D_THETA = 28      # state, workspace, tridiagonal coefficients, scratch
FIELDS_3D_SPLIT = 30      # plus the barotropic workspace
FIELDS_2D = 12            # free-surface and depth-integrated fields


def memory_model(grid: CGrid, scheme: str, itemsize: int = 8) -> dict[str, Any]:
    """Bytes the working set needs, and what that implies for a 32 GB device."""
    cells = grid.nx * grid.ny * grid.nz
    n3 = FIELDS_3D_SPLIT if scheme == "split_explicit" else FIELDS_3D_THETA
    total = (n3 * cells + FIELDS_2D * grid.nx * grid.ny) * itemsize
    per_cell = total / cells
    return {
        "cells": cells,
        "fields_3d": n3,
        "bytes": int(total),
        "bytes_per_cell": per_cell,
        "gib": total / 2**30,
        # What resolution actually fits, which is often the binding constraint
        # on a GPU rather than speed.
        "max_cells_32gib": int(32 * 2**30 / per_cell),
        "max_nx_32gib_nz30": int(np.sqrt(32 * 2**30 / per_cell / 30)),
    }
