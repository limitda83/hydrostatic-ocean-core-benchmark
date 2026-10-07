#########################################################################
#  Module: vertical_var                                                 #
#  Description: Vertical operators for a layer thickness that varies    #
#               with column and time (docs/03_discretization_spec.md    #
#               S10.3). These are the S7.3-S7.4 operators with the      #
#               scalar dz replaced by dz3[k,j,i]; with a uniform dz3    #
#               they reduce to the same numbers (verification V5-1).    #
#  Pipeline: domain -> vertical_var -> model3d_v05                      #
#########################################################################

from __future__ import annotations

import numpy as np


def _safe(dz: np.ndarray) -> np.ndarray:
    """1/dz that is zero in dry cells instead of infinite."""
    return np.where(dz > 0.0, 1.0 / np.where(dz > 0.0, dz, 1.0), 0.0)


def interface_thickness(dz3: np.ndarray) -> np.ndarray:
    """Distance between adjacent layer centres, at interfaces [nz+1,ny,nx].

    The end interfaces are never divided by (their fluxes are prescribed),
    so they are filled with the adjacent layer thickness to stay finite.
    """
    nz = dz3.shape[0]
    dzi = np.empty((nz + 1,) + dz3.shape[1:], dtype=dz3.dtype)
    dzi[0] = dz3[0]
    dzi[1:nz] = 0.5 * (dz3[:nz - 1] + dz3[1:nz])
    dzi[nz] = dz3[nz - 1]
    return dzi


def diffusion_coeffs_var(nu: np.ndarray, dz3: np.ndarray, dt: float,
                         theta_v: float, bottom_drag: float = 0.0,
                         mask3: np.ndarray | None = None
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tridiagonal coefficients of A = I - theta_v dt Dz on a variable grid.

    A dry cell gets the identity row, so the batched Thomas sweep can run
    over the full [nz,ny,nx] block without branching - which is what keeps
    the GPU kernel of S7.4 a single dense launch.
    """
    nz = nu.shape[0] - 1
    dzi = interface_thickness(dz3)
    inv_dz = _safe(dz3)
    fac = theta_v * dt * inv_dz
    alpha_top = fac * nu[:nz] * _safe(dzi[:nz])
    alpha_bot = fac * nu[1:nz + 1] * _safe(dzi[1:nz + 1])

    sub = -alpha_top.copy()
    sup = -alpha_bot.copy()
    diag = 1.0 + alpha_top + alpha_bot
    # Surface: prescribed stress, no diffusive coupling through interface 0.
    diag[0] = 1.0 + alpha_bot[0]
    sub[0] = 0.0
    # Bottom of each column: prescribed stress plus optional linear drag. With
    # partial cells the bottom is column-dependent, so it is found from mask3.
    if mask3 is None:
        bottom = np.zeros(dz3.shape, dtype=bool)
        bottom[nz - 1] = True
    else:
        below = np.concatenate([mask3[1:], np.zeros((1,) + mask3.shape[1:])],
                               axis=0)
        bottom = (mask3 > 0) & (below == 0)
    diag = np.where(bottom, 1.0 + alpha_top + theta_v * dt * bottom_drag * inv_dz,
                    diag)
    sup = np.where(bottom, 0.0, sup)
    if mask3 is not None:
        dry = mask3 == 0
        diag = np.where(dry, 1.0, diag)
        sub = np.where(dry, 0.0, sub)
        sup = np.where(dry, 0.0, sup)
        # A wet cell must not couple upward into a dry one (never happens for
        # a top-down filled column) nor downward past the bottom (handled).
    return sub, diag, sup


def apply_diffusion_var(field: np.ndarray, nu: np.ndarray, dz3: np.ndarray,
                        surface_flux: np.ndarray | None = None,
                        bottom_drag: float = 0.0,
                        mask3: np.ndarray | None = None) -> np.ndarray:
    """Explicit Dz[field] on the variable grid, same boundary treatment."""
    nz = field.shape[0]
    dzi = interface_thickness(dz3)
    flux = np.zeros((nz + 1,) + field.shape[1:], dtype=field.dtype)
    flux[1:nz] = nu[1:nz] * (field[:nz - 1] - field[1:nz]) * _safe(dzi[1:nz])
    flux[0] = 0.0 if surface_flux is None else surface_flux
    if mask3 is None:
        flux[nz] = bottom_drag * field[nz - 1]
    else:
        below = np.concatenate([mask3[1:], np.zeros((1,) + mask3.shape[1:])],
                               axis=0)
        bottom = (mask3 > 0) & (below == 0)
        # No diffusive flux across a bottom interface; drag acts on the cell.
        # The drag enters ONLY through the per-cell term below: writing it
        # into flux[nz] as well counted it twice on every full-depth column.
        # (Every verified case runs with bottom_drag = 0, so no recorded
        # result changes; the compiled backend mirrors this form.)
        flux[1:nz] = np.where(bottom[:nz - 1], 0.0, flux[1:nz])
        flux[nz] = 0.0
        drag = bottom_drag * field * bottom
        out = (flux[:nz] - flux[1:nz + 1]) * _safe(dz3)
        return (out - drag * _safe(dz3)) * (mask3 > 0)
    return (flux[:nz] - flux[1:nz + 1]) * _safe(dz3)


def w_from_transport_divergence(trans_div: np.ndarray) -> np.ndarray:
    """w at interfaces from the LAYER TRANSPORT divergence, w = 0 at the bottom.

    The transport divergence is d(dz*u)/dx + d(dz*v)/dy, which is not
    dz * div(u,v) once the layer thickness varies horizontally. Building w
    from the latter breaks constancy preservation: an initially uniform
    tracer picked up a 0.26 K spread in sixty steps before this distinction
    was made.
    """
    nz = trans_div.shape[0]
    w = np.zeros((nz + 1,) + trans_div.shape[1:], dtype=trans_div.dtype)
    w[:nz] = -np.cumsum(trans_div[::-1], axis=0)[::-1]
    return w


def w_from_divergence_var(div: np.ndarray, dz3: np.ndarray) -> np.ndarray:
    """w at interfaces from the horizontal divergence, w = 0 at the bottom."""
    nz = div.shape[0]
    w = np.zeros((nz + 1,) + div.shape[1:], dtype=div.dtype)
    w[:nz] = -np.cumsum((dz3 * div)[::-1], axis=0)[::-1]
    return w


def buoyancy_potential_var(b: np.ndarray, dz3: np.ndarray) -> np.ndarray:
    """Phi[k] = int_z^0 b dz' on the variable grid."""
    above = np.concatenate([np.zeros((1,) + b.shape[1:], dtype=b.dtype),
                            np.cumsum((dz3 * b)[:-1], axis=0)], axis=0)
    return 0.5 * dz3 * b + above


def remove_depth_mean_var(phi: np.ndarray, dz3: np.ndarray) -> np.ndarray:
    """Phi' = Phi - <Phi>_z with the depth mean weighted by real thickness
    (spec S10.7). Using an unweighted mean over a partial-cell column leaves
    a spurious barotropic forcing proportional to the partial-cell fraction."""
    total = np.sum(dz3, axis=0, keepdims=True)
    mean = np.sum(dz3 * phi, axis=0, keepdims=True) * _safe(total)
    return phi - mean


def w_at_centres_var(w_interface: np.ndarray) -> np.ndarray:
    return 0.5 * (w_interface[:-1] + w_interface[1:])
