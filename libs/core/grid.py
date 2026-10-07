#########################################################################
#  Module: grid                                                         #
#  Description: Arakawa C-grid geometry for the doubly periodic 2D      #
#               barotropic domain (docs/03_discretization_spec.md S2).  #
#  Pipeline: config -> grid -> operators / cases / schemes              #
#########################################################################

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from libs.utils.config import Config


@dataclass(frozen=True)
class CGrid:
    """Uniform Arakawa C-grid.

    Array layout is [j, i] = (y, x). Variable staggering:
        eta[j, i] at cell centre  ( (i+1/2)dx , (j+1/2)dy )
        u  [j, i] at east face    ( (i+1)dx   , (j+1/2)dy )
        v  [j, i] at north face   ( (i+1/2)dx , (j+1)dy   )
    """

    nx: int
    ny: int
    Lx: float
    Ly: float
    bc_x: str = "periodic"
    bc_y: str = "periodic"
    nz: int = 1          # vertical layers; 1 means the 2D barotropic model
    depth: float = 0.0   # resting depth H [m]; 0 means "unused (2D)"

    @property
    def dx(self) -> float:
        return self.Lx / self.nx

    @property
    def dy(self) -> float:
        return self.Ly / self.ny

    @property
    def cell_area(self) -> float:
        return self.dx * self.dy

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    @property
    def shape3d(self) -> tuple[int, int, int]:
        return (self.nz, self.ny, self.nx)

    @property
    def dz(self) -> float:
        return self.depth / self.nz

    def z_centre(self) -> np.ndarray:
        """Height above the bottom at layer centres, zt = z + H (spec S7.3)."""
        k = np.arange(self.nz, dtype=np.float64)
        return (self.nz - k - 0.5) * self.dz

    def z_interface(self) -> np.ndarray:
        """Height above the bottom at interfaces, k = 0 (surface) .. nz."""
        k = np.arange(self.nz + 1, dtype=np.float64)
        return (self.nz - k) * self.dz

    def coords(self, variable: str) -> tuple[np.ndarray, np.ndarray]:
        """Physical (x, y) coordinate arrays for 'eta', 'u' or 'v' points."""
        i = np.arange(self.nx, dtype=np.float64)
        j = np.arange(self.ny, dtype=np.float64)
        offsets = {"eta": (0.5, 0.5), "u": (1.0, 0.5), "v": (0.5, 1.0)}
        if variable not in offsets:
            raise ValueError(f"unknown variable '{variable}' (expected eta|u|v)")
        ox, oy = offsets[variable]
        x = (i + ox) * self.dx
        y = (j + oy) * self.dy
        return np.meshgrid(x, y, indexing="xy")   # -> arrays shaped [ny, nx]

    def zeros(self, dtype: np.dtype | type = np.float64) -> np.ndarray:
        return np.zeros(self.shape, dtype=dtype)


def build_grid(config: Config, nx: int | None = None, ny: int | None = None,
               nz: int | None = None) -> CGrid:
    """Construct the grid from config, optionally overriding the resolution."""
    grid = CGrid(
        nx=int(nx if nx is not None else config.get("grid.nx")),
        ny=int(ny if ny is not None else config.get("grid.ny")),
        Lx=float(config.get("grid.Lx")),
        Ly=float(config.get("grid.Ly")),
        bc_x=str(config.get("grid.bc_x")),
        bc_y=str(config.get("grid.bc_y")),
        nz=int(nz if nz is not None else config.get("grid.nz", 1)),
        depth=float(config.get("physics.H")),
    )
    if grid.bc_x != "periodic" or grid.bc_y != "periodic":
        raise NotImplementedError(
            "spec v0.1 (Phase 0) implements doubly periodic boundaries only; "
            "wall boundaries arrive with the Kelvin-wave case in Phase 1"
        )
    return grid
