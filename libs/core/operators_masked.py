#########################################################################
#  Module: operators_masked                                             #
#  Description: C-grid operators on a masked domain (docs/03_           #
#               discretization_spec.md S10.4). Every horizontal          #
#               operator is the S2.1 operator multiplied by the face     #
#               mask, which is what makes D and -G adjoint and keeps the #
#               free-surface operator symmetric positive definite.       #
#  Pipeline: domain -> operators_masked -> model3d_v05                  #
#########################################################################

from __future__ import annotations

import numpy as np

from libs.core.grid import CGrid


def gradx_u_m(eta: np.ndarray, grid: CGrid, mask_u: np.ndarray) -> np.ndarray:
    """Gx[eta] at u points, zero across a dry face (a wall)."""
    return mask_u * (np.roll(eta, -1, axis=-1) - eta) / grid.dx


def grady_v_m(eta: np.ndarray, grid: CGrid, mask_v: np.ndarray) -> np.ndarray:
    return mask_v * (np.roll(eta, -1, axis=-2) - eta) / grid.dy


def div_m(u: np.ndarray, v: np.ndarray, grid: CGrid, mask: np.ndarray
          ) -> np.ndarray:
    """Divergence at eta points. u and v must already carry their face masks;
    the wall condition is then no-normal-flow exactly, not approximately."""
    return mask * ((u - np.roll(u, 1, axis=-1)) / grid.dx
                   + (v - np.roll(v, 1, axis=-2)) / grid.dy)


def _masked_average(field: np.ndarray, weights: list[tuple[np.ndarray, np.ndarray]]
                    ) -> np.ndarray:
    """Average `field` over the listed (shifted value, shifted mask) pairs,
    normalised by the wet weight so a land neighbour does not dilute the
    result towards zero."""
    num = sum(w * f for f, w in weights)
    den = sum(w for _, w in weights)
    return np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0), 0.0)


def avg_v_to_u_m(v: np.ndarray, mask_v: np.ndarray, mask_u: np.ndarray
                 ) -> np.ndarray:
    """Coriolis average of v onto u points over wet v faces only."""
    vs, ms = np.roll(v, 1, axis=-2), np.roll(mask_v, 1, axis=-2)
    out = _masked_average(v, [(v, mask_v), (vs, ms),
                              (np.roll(v, -1, axis=-1), np.roll(mask_v, -1, axis=-1)),
                              (np.roll(vs, -1, axis=-1), np.roll(ms, -1, axis=-1))])
    return out * mask_u


def avg_u_to_v_m(u: np.ndarray, mask_u: np.ndarray, mask_v: np.ndarray
                 ) -> np.ndarray:
    uw, mw = np.roll(u, 1, axis=-1), np.roll(mask_u, 1, axis=-1)
    out = _masked_average(u, [(u, mask_u), (uw, mw),
                              (np.roll(u, -1, axis=-2), np.roll(mask_u, -1, axis=-2)),
                              (np.roll(uw, -1, axis=-2), np.roll(mw, -1, axis=-2))])
    return out * mask_v


def laplacian_h_m(field: np.ndarray, grid: CGrid, mask_u: np.ndarray,
                  mask_v: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Horizontal Laplacian as D o G, so it stays negative semi-definite."""
    return div_m(gradx_u_m(field, grid, mask_u),
                 grady_v_m(field, grid, mask_v), grid, mask)
