#########################################################################
#  Module: domain                                                       #
#  Description: Everything the v0.5 core needs that the flat-bottom     #
#               doubly periodic core did not: bathymetry, land and face #
#               masks, the vertical coordinate (z-level with partial    #
#               cells, z*, sigma) and the boundary condition kind.      #
#               docs/03_discretization_spec.md S10.2-S10.4.             #
#  Pipeline: grid + bathymetry -> domain -> operators_masked/model3d_v05#
#########################################################################

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from libs.core.bathymetry import Roughness, bathymetry_from_config
from libs.core.grid import CGrid

VCOORDS = ("zlevel", "zstar", "sigma")
BCS = ("periodic", "closed", "open")


@dataclass
class Domain:
    """Grid geometry plus bathymetry, masks and vertical coordinate.

    All fields are [ny, nx] unless noted. Face depths use the minimum rule
    of spec S10.2 so that a face never carries more water than the shallower
    of the two cells it joins.
    """

    grid: CGrid
    H: np.ndarray                  # bathymetry at eta points [m], positive down
    mask: np.ndarray               # 1 = ocean, 0 = land
    vcoord: str = "zlevel"
    bc_x: str = "periodic"
    bc_y: str = "periodic"
    min_partial: float = 0.1       # thinnest partial cell, as a fraction of dzr
    # How a face depth is taken from its two cells (spec S10.2).
    #   "min"       the production rule: a face never carries more water than
    #               the shallower cell holds. Only FIRST order on a smooth H
    #               (measured 1.04, V5-2).
    #   "mean"      second order (measured 2.00) but can over-fill a face next
    #               to a much shallower cell.
    #   "harmonic"  second order as well; damps the transition more strongly.
    face_rule: str = "min"

    # Derived; filled by __post_init__.
    mask_u: np.ndarray = field(init=False)
    mask_v: np.ndarray = field(init=False)
    Hu: np.ndarray = field(init=False)
    Hv: np.ndarray = field(init=False)
    dzr: np.ndarray = field(init=False)      # reference thickness [nz]
    dz3_ref: np.ndarray = field(init=False)  # [nz,ny,nx] resting thickness
    mask3: np.ndarray = field(init=False)    # [nz,ny,nx] wet cells
    mask3u: np.ndarray = field(init=False)   # [nz,ny,nx] wet u faces
    mask3v: np.ndarray = field(init=False)   # [nz,ny,nx] wet v faces

    def __post_init__(self) -> None:
        if self.vcoord not in VCOORDS:
            raise ValueError(f"vcoord must be one of {VCOORDS}")
        for name, bc in (("bc_x", self.bc_x), ("bc_y", self.bc_y)):
            if bc not in BCS:
                raise ValueError(f"{name} must be one of {BCS}")
        if np.any(self.H[self.mask > 0] <= 0.0):
            raise ValueError("bathymetry must be positive in every wet cell")

        g = self.grid
        # Face masks: a face is wet only if both neighbours are wet, and a
        # domain-edge face is dry unless that direction is periodic.
        mu = self.mask * np.roll(self.mask, -1, axis=-1)
        mv = self.mask * np.roll(self.mask, -1, axis=-2)
        if self.bc_x != "periodic":
            mu[:, -1] = 0.0
        if self.bc_y != "periodic":
            mv[-1, :] = 0.0
        self.mask_u, self.mask_v = mu, mv

        self.Hu = mu * self._face(self.H, -1)
        self.Hv = mv * self._face(self.H, -2)

        # Reference level thicknesses. The table has to span the DEEPEST
        # column, not the nominal depth, or every column deeper than
        # physics.H would silently lose its bottom (measured: 227 m of a
        # 1270 m column for bathymetry.kind='rough').
        h_max = float(np.max(self.H[self.mask > 0]))
        self.dzr = np.full(g.nz, h_max / g.nz)
        self.dz3_ref, self.mask3 = self._resting_thickness()
        # 3D face masks. Without these a horizontal gradient taken across a
        # bottom step reads the dry cell's zero as if it were water: the
        # seamount test then reports metres per second of spurious flow even
        # on a z-level grid, where the answer must be zero.
        self.mask3u = (self.mask3 * np.roll(self.mask3, -1, axis=-1)
                       * self.mask_u[None, :, :])
        self.mask3v = (self.mask3 * np.roll(self.mask3, -1, axis=-2)
                       * self.mask_v[None, :, :])

    def _face(self, a: np.ndarray, axis: int) -> np.ndarray:
        """Combine a cell-centred field onto the faces along `axis`."""
        b = np.roll(a, -1, axis=axis)
        if self.face_rule == "min":
            return np.minimum(a, b)
        if self.face_rule == "mean":
            return 0.5 * (a + b)
        if self.face_rule == "harmonic":
            tot = a + b
            return np.where(tot > 0.0, 2.0 * a * b / np.where(tot > 0.0, tot, 1.0), 0.0)
        raise ValueError(f"face_rule must be min|mean|harmonic, got "
                         f"'{self.face_rule}'")

    # ------------------------------------------------------------ geometry
    def _resting_thickness(self) -> tuple[np.ndarray, np.ndarray]:
        """Resting layer thickness and wet mask (spec S10.3)."""
        nz, ny, nx = self.grid.shape3d
        if self.vcoord == "sigma":
            s = self.dzr / self.dzr.sum()
            dz3 = s[:, None, None] * self.H[None, :, :]
            m3 = np.broadcast_to(self.mask, (nz, ny, nx)).astype(np.float64)
            return dz3 * m3, m3.copy()

        # z-level: fill fixed levels top-down, the last wet cell is partial.
        edges = np.concatenate([[0.0], np.cumsum(self.dzr)])   # depth of faces
        remaining = self.H[None, :, :] - edges[:-1, None, None]
        dz3 = np.clip(remaining, 0.0, self.dzr[:, None, None])
        thin = (dz3 > 0.0) & (dz3 < self.min_partial * self.dzr[:, None, None])
        dz3 = np.where(thin, 0.0, dz3)
        m3 = (dz3 > 0.0).astype(np.float64) * self.mask[None, :, :]
        return dz3 * m3, m3

    def layer_thickness(self, eta: np.ndarray) -> np.ndarray:
        """dz3[k,j,i] for the current free surface (spec S10.3).

        zlevel keeps the linear free surface of v0.4 (thickness independent
        of eta), which is what makes the V5-1 reduction test exact.
        """
        if self.vcoord == "zlevel":
            return self.dz3_ref
        scale = np.where(self.mask > 0, 1.0 + eta / np.where(self.H > 0, self.H, 1.0), 1.0)
        return self.dz3_ref * scale[None, :, :]

    def face_thickness(self, dz3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Layer thickness at u and v faces: the minimum rule of S10.2."""
        dzu = self.mask3u * self._face(dz3, -1)
        dzv = self.mask3v * self._face(dz3, -2)
        return dzu, dzv

    def depth_at_centres(self, eta: np.ndarray | None = None) -> np.ndarray:
        """Depth of layer centres, positive downward [nz,ny,nx]. The EOS needs
        it for the thermobaric and compressibility terms (spec S10.6)."""
        dz3 = self.dz3_ref if eta is None else self.layer_thickness(eta)
        above = np.concatenate([np.zeros((1,) + dz3.shape[1:]),
                                np.cumsum(dz3[:-1], axis=0)], axis=0)
        return above + 0.5 * dz3

    # ------------------------------------------------------------ reporting
    @property
    def roughness(self) -> Roughness:
        """Roughness over wet faces only, so a closed boundary's wrap-around
        jump between the two domain edges is not counted as topography."""
        wet = self.mask > 0
        h = self.H[wet]
        rx = 0.0
        for axis, fm in ((-1, self.mask_u), (-2, self.mask_v)):
            a, b = self.H, np.roll(self.H, -1, axis=axis)
            sel = fm > 0
            if np.any(sel):
                rx = max(rx, float(np.max(np.abs(a - b)[sel] / (a + b)[sel])))
        return Roughness(r_std=float(np.std(h) / np.mean(h)), rx0=rx,
                         h_ratio=float(np.max(h) / np.min(h)))

    def depth_error(self) -> float:
        """Largest |sum_k dz3 - H| over wet columns. Zero for sigma; for
        z-level it is the staircase/partial-cell truncation, bounded by
        min_partial * dzr, and is a property of the coordinate, not a bug."""
        col = self.dz3_ref.sum(axis=0)
        wet = self.mask > 0
        return float(np.max(np.abs(col - self.H)[wet])) if np.any(wet) else 0.0

    @property
    def is_flat_periodic(self) -> bool:
        """True when the domain reduces to the v0.4 configuration (V5-1)."""
        return (self.vcoord == "zlevel" and self.bc_x == "periodic"
                and self.bc_y == "periodic" and np.all(self.mask == 1.0)
                and float(np.ptp(self.H)) == 0.0)

    def summary(self) -> dict:
        r = self.roughness
        return {"vcoord": self.vcoord, "bc_x": self.bc_x, "bc_y": self.bc_y,
                "face_rule": self.face_rule,
                "H_min": float(np.min(self.H[self.mask > 0])),
                "H_max": float(np.max(self.H[self.mask > 0])),
                "wet_fraction": float(np.mean(self.mask3)),
                "depth_error_max": self.depth_error(),
                **r.as_dict()}


def build_domain(config, grid: CGrid) -> Domain:
    """Assemble the domain from config (R3)."""
    H, mask = bathymetry_from_config(grid, config)
    return Domain(grid=grid, H=H, mask=mask,
                  vcoord=str(config.get("domain.vcoord", "zlevel")),
                  bc_x=str(config.get("domain.bc_x", "periodic")),
                  bc_y=str(config.get("domain.bc_y", "periodic")),
                  min_partial=float(config.get("domain.min_partial", 0.1)),
                  face_rule=str(config.get("domain.face_rule", "min")))
