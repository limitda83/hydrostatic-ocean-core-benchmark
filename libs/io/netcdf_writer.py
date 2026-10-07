#########################################################################
#  Module: netcdf_writer                                                #
#  Description: CF-conventions NetCDF4 output for the C-grid state.     #
#               netCDF4 is the primary I/O library for this project.    #
#  Pipeline: driver -> netcdf_writer -> output/<run_id>/state.nc        #
#########################################################################

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from libs.core.grid import CGrid
from libs.core.schemes import State

LOGGER = logging.getLogger(__name__)


def write_state(path: Path, grid: CGrid, numeric: State, exact: State | None,
                fmt: str = "NETCDF4", complevel: int = 4) -> Path:
    """Write the numerical (and optionally exact) state with CF attributes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(path, "w", format=fmt) as ds:
        ds.Conventions = "CF-1.10"
        ds.title = "cfd_exp 2D linear rotating shallow water state"
        ds.source = "cfd_exp reference backend (NumPy fp64)"
        ds.institution = "cfd_exp"
        ds.comment = ("Arakawa C-grid; eta at cell centres, u at east faces, "
                      "v at north faces; see docs/03_discretization_spec.md")
        ds.time_coverage_end = f"{numeric.t:.6f} s"

        ds.createDimension("y", grid.ny)
        ds.createDimension("x", grid.nx)

        x_e, y_e = grid.coords("eta")
        _coord(ds, "x", x_e[0, :], "projection_x_coordinate", "cell centre x")
        _coord(ds, "y", y_e[:, 0], "projection_y_coordinate", "cell centre y")

        fields = {
            "eta": (numeric.eta, "m", "sea_surface_height_above_geoid",
                    "free surface elevation"),
            "u": (numeric.u, "m s-1", "sea_water_x_velocity",
                  "zonal velocity at east faces"),
            "v": (numeric.v, "m s-1", "sea_water_y_velocity",
                  "meridional velocity at north faces"),
        }
        for name, (data, units, standard, long_name) in fields.items():
            _field(ds, name, data, units, standard, long_name, complevel)
            if exact is not None:
                ref = {"eta": exact.eta, "u": exact.u, "v": exact.v}[name]
                _field(ds, f"{name}_exact", ref, units, standard,
                       f"{long_name} (exact solution)", complevel)
                _field(ds, f"{name}_error", data - ref, units, "",
                       f"{long_name} error (numeric - exact)", complevel)

    LOGGER.info(f"state written: {path}")
    return path


def _coord(ds: Dataset, name: str, values: np.ndarray, standard: str,
           long_name: str) -> None:
    var = ds.createVariable(name, "f8", (name,))
    var.units = "m"
    var.standard_name = standard
    var.long_name = long_name
    var.axis = name.upper()
    var[:] = values


def _field(ds: Dataset, name: str, data: np.ndarray, units: str, standard: str,
           long_name: str, complevel: int) -> None:
    var = ds.createVariable(name, "f8", ("y", "x"), zlib=complevel > 0,
                            complevel=complevel)
    var.units = units
    if standard:
        var.standard_name = standard
    var.long_name = long_name
    var.coordinates = "y x"
    var[:, :] = data
