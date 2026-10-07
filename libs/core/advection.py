#########################################################################
#  Module: advection                                                    #
#  Description: Flux-form advection and horizontal diffusion of         #
#               docs/03_discretization_spec.md S9.2-S9.3. Tracer        #
#               advection carries the scheme choice (axis C) because    #
#               that is where the scheme drives spurious diapycnal      #
#               mixing (literature L6); momentum keeps the second-order #
#               centred flux form in both settings.                     #
#  Pipeline: grid/vertical -> advection -> model3d                      #
#########################################################################

from __future__ import annotations

import numpy as np

from libs.core.grid import CGrid


def _face(c: np.ndarray, vel: np.ndarray, axis: int, scheme: str) -> np.ndarray:
    """Tracer value on the downstream face, by scheme."""
    c_next = np.roll(c, -1, axis=axis)
    if scheme == "centered2":
        return 0.5 * (c + c_next)
    if scheme == "upwind1":
        return np.where(vel > 0.0, c, c_next)
    raise ValueError(f"unknown advection scheme '{scheme}'")


def _face_z(c: np.ndarray, w: np.ndarray, scheme: str) -> np.ndarray:
    """Tracer on vertical interfaces 1..nz-1.

    Interface k sits between layer k-1 (above) and layer k (below); w is
    positive upward, so an upward flux carries the value from below.
    """
    above, below = c[:-1], c[1:]
    if scheme == "centered2":
        return 0.5 * (above + below)
    return np.where(w[1:-1] > 0.0, below, above)


def tracer_advection(c: np.ndarray, u: np.ndarray, v: np.ndarray, w: np.ndarray,
                     grid: CGrid, scheme: str) -> np.ndarray:
    """DIV(u C) in flux form (spec S9.2).

    The surface advective flux is set to zero, which is the standard linearised
    free-surface treatment: w at the surface is d(eta)/dt, and the tracer flux
    that goes with it belongs to the free-surface term rather than here.
    """
    nz = c.shape[0]
    fx = u * _face(c, u, -1, scheme)
    fy = v * _face(c, v, -2, scheme)

    fz = np.zeros((nz + 1,) + c.shape[1:], dtype=c.dtype)
    if nz > 1:
        fz[1:nz] = w[1:nz] * _face_z(c, w, scheme)

    return ((fx - np.roll(fx, 1, axis=-1)) / grid.dx
            + (fy - np.roll(fy, 1, axis=-2)) / grid.dy
            + (fz[:nz] - fz[1:nz + 1]) / grid.dz)


def momentum_advection(u: np.ndarray, v: np.ndarray, w: np.ndarray,
                       grid: CGrid) -> tuple[np.ndarray, np.ndarray]:
    """Second-order centred flux-form advection of u and v (spec S9.2)."""
    nz = u.shape[0]

    # --- u at east faces -------------------------------------------------
    ubar_c = 0.5 * (np.roll(u, 1, axis=-1) + u)              # cell centres
    adv_ux = (np.roll(ubar_c, -1, axis=-1) ** 2 - ubar_c ** 2) / grid.dx
    vbar = 0.5 * (v + np.roll(v, -1, axis=-1))               # corners (i+1/2, j+1/2)
    ubar_y = 0.5 * (u + np.roll(u, -1, axis=-2))
    fy_u = vbar * ubar_y
    adv_uy = (fy_u - np.roll(fy_u, 1, axis=-2)) / grid.dy

    wbar_u = 0.5 * (w + np.roll(w, -1, axis=-1))             # (i+1/2, interfaces)
    fz_u = np.zeros_like(wbar_u)
    if nz > 1:
        fz_u[1:nz] = wbar_u[1:nz] * 0.5 * (u[:-1] + u[1:])
    adv_uz = (fz_u[:nz] - fz_u[1:nz + 1]) / grid.dz

    # --- v at north faces ------------------------------------------------
    vbar_c = 0.5 * (np.roll(v, 1, axis=-2) + v)
    adv_vy = (np.roll(vbar_c, -1, axis=-2) ** 2 - vbar_c ** 2) / grid.dy
    ubar = 0.5 * (u + np.roll(u, -1, axis=-2))
    vbar_x = 0.5 * (v + np.roll(v, -1, axis=-1))
    fx_v = ubar * vbar_x
    adv_vx = (fx_v - np.roll(fx_v, 1, axis=-1)) / grid.dx

    wbar_v = 0.5 * (w + np.roll(w, -1, axis=-2))
    fz_v = np.zeros_like(wbar_v)
    if nz > 1:
        fz_v[1:nz] = wbar_v[1:nz] * 0.5 * (v[:-1] + v[1:])
    adv_vz = (fz_v[:nz] - fz_v[1:nz + 1]) / grid.dz

    return adv_ux + adv_uy + adv_uz, adv_vx + adv_vy + adv_vz


def laplacian_h(a: np.ndarray, grid: CGrid) -> np.ndarray:
    """Horizontal Laplacian, applied explicitly (spec S9.3).

    Explicit treatment needs dt < dx^2/(4 A_h); the horizontal grid is coarse
    enough that this is far looser than the vertical diffusion limit, which is
    why only the vertical operator is made implicit.
    """
    return ((np.roll(a, -1, axis=-1) - 2.0 * a + np.roll(a, 1, axis=-1)) / grid.dx**2
            + (np.roll(a, -1, axis=-2) - 2.0 * a + np.roll(a, 1, axis=-2)) / grid.dy**2)
