#########################################################################
#  Module: operators                                                    #
#  Description: Discrete C-grid operators of docs/03_discretization_    #
#               spec.md S2.1. Pure functions (no in-place writes) so    #
#               the same source ports to JAX in Phase 6.                #
#                                                                       #
#  Axes are addressed as -1 (x) and -2 (y), so every operator applies   #
#  unchanged to a 2D field [j,i] and to a 3D field [k,j,i] level by     #
#  level. The 2D verification therefore also validates the 3D horizontal#
#  operators - they are the same code, not a parallel copy.             #
#  Pipeline: grid -> operators -> schemes / cases / diagnostics         #
#########################################################################

from __future__ import annotations

import numpy as np

from libs.core.grid import CGrid

# All operators assume doubly periodic wrap-around, expressed as index rolls
# rather than halo exchange (spec S6.4).


def gradx_u(eta: np.ndarray, grid: CGrid) -> np.ndarray:
    """Gx[eta] = (eta[j,i+1] - eta[j,i]) / dx, evaluated at u points."""
    return (np.roll(eta, -1, axis=-1) - eta) / grid.dx


def grady_v(eta: np.ndarray, grid: CGrid) -> np.ndarray:
    """Gy[eta] = (eta[j+1,i] - eta[j,i]) / dy, evaluated at v points."""
    return (np.roll(eta, -1, axis=-2) - eta) / grid.dy


def div_eta(u: np.ndarray, v: np.ndarray, grid: CGrid) -> np.ndarray:
    """D[u,v] at eta points."""
    return (u - np.roll(u, 1, axis=-1)) / grid.dx + (v - np.roll(v, 1, axis=-2)) / grid.dy


def avg_v_to_u(v: np.ndarray) -> np.ndarray:
    """A_u[v] = 1/4 ( v[j,i] + v[j-1,i] + v[j,i+1] + v[j-1,i+1] )."""
    v_south = np.roll(v, 1, axis=-2)
    return 0.25 * (v + v_south + np.roll(v, -1, axis=-1) + np.roll(v_south, -1, axis=-1))


def avg_u_to_v(u: np.ndarray) -> np.ndarray:
    """A_v[u] = 1/4 ( u[j,i] + u[j,i-1] + u[j+1,i] + u[j+1,i-1] )."""
    u_west = np.roll(u, 1, axis=-1)
    return 0.25 * (u + u_west + np.roll(u, -1, axis=-2) + np.roll(u_west, -1, axis=-2))


def laplacian_eta(eta: np.ndarray, grid: CGrid) -> np.ndarray:
    """L[eta] = D o G [eta]. Defined as the composition, never as a separate
    5-point stencil, so that (I - c L) stays symmetric positive definite."""
    return div_eta(gradx_u(eta, grid), grady_v(eta, grid), grid)
