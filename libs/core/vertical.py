#########################################################################
#  Module: vertical                                                     #
#  Description: Vertical operators and the batched tridiagonal solve of #
#               docs/03_discretization_spec.md S7.3-S7.4. This is the   #
#               kernel that maps best to a GPU - nx*ny independent      #
#               systems of size nz - and the reason semi-implicit ocean #
#               models exist at all (it removes the dt < dz^2/2nu       #
#               stability limit).                                       #
#  Pipeline: grid -> vertical -> model3d                                #
#########################################################################

from __future__ import annotations

import numpy as np


def thomas_batched(sub: np.ndarray, diag: np.ndarray, sup: np.ndarray,
                   rhs: np.ndarray) -> np.ndarray:
    """Solve one tridiagonal system per column, vectorised over [ny, nx].

    All coefficient arrays are full [nz, ny, nx] even when the values happen
    to be column-independent. Factorising once for a uniform matrix would
    understate the cost of the real case (turbulence closures give a
    different matrix per column), and the point of this kernel is to measure
    that cost honestly (spec S7.4).

    sub[0] and sup[nz-1] are ignored.
    """
    nz = diag.shape[0]
    c_star = np.empty_like(sup)
    d_star = np.empty_like(rhs)

    c_star[0] = sup[0] / diag[0]
    d_star[0] = rhs[0] / diag[0]
    for k in range(1, nz):
        denom = diag[k] - sub[k] * c_star[k - 1]
        c_star[k] = sup[k] / denom
        d_star[k] = (rhs[k] - sub[k] * d_star[k - 1]) / denom

    x = np.empty_like(rhs)
    x[nz - 1] = d_star[nz - 1]
    for k in range(nz - 2, -1, -1):
        x[k] = d_star[k] - c_star[k] * x[k + 1]
    return x


def diffusion_coeffs(nu: np.ndarray, dz: float, dt: float, theta_v: float,
                     bottom_drag: float = 0.0
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tridiagonal coefficients of A = I - theta_v*dt*Dz (spec S7.4).

    ``nu`` is the viscosity at layer interfaces, shape [nz+1, ny, nx];
    nu[0] is the surface interface and nu[nz] the bottom. Flux boundary
    conditions are imposed by dropping the corresponding off-diagonal term.
    """
    nz = nu.shape[0] - 1
    fac = theta_v * dt / dz**2
    alpha_top = fac * nu[:nz]        # interface above layer k
    alpha_bot = fac * nu[1:nz + 1]   # interface below layer k

    sub = -alpha_top.copy()
    sup = -alpha_bot.copy()
    diag = 1.0 + alpha_top + alpha_bot

    # Surface: prescribed stress, so no diffusive coupling through interface 0.
    diag[0] = 1.0 + alpha_bot[0]
    sub[0] = 0.0
    # Bottom: prescribed stress plus optional linear drag.
    diag[nz - 1] = 1.0 + alpha_top[nz - 1] + theta_v * dt * bottom_drag / dz
    sup[nz - 1] = 0.0
    return sub, diag, sup


def apply_diffusion(field: np.ndarray, nu: np.ndarray, dz: float,
                    surface_flux: np.ndarray | None = None,
                    bottom_drag: float = 0.0) -> np.ndarray:
    """Explicit Dz[field] = d/dz( nu d(field)/dz ), same boundary treatment."""
    nz = field.shape[0]
    flux = np.zeros((nz + 1,) + field.shape[1:], dtype=field.dtype)
    # Interior interfaces: positive flux is upward gradient nu*(a[k-1]-a[k])/dz.
    flux[1:nz] = nu[1:nz] * (field[:nz - 1] - field[1:nz]) / dz
    flux[0] = 0.0 if surface_flux is None else surface_flux
    flux[nz] = bottom_drag * field[nz - 1]
    return (flux[:nz] - flux[1:nz + 1]) / dz


def w_from_divergence(div: np.ndarray, dz: float) -> np.ndarray:
    """w at interfaces from the horizontal divergence, integrating up from the
    bottom where w = 0 (spec S7.3). Returns shape [nz+1, ny, nx]."""
    nz = div.shape[0]
    w = np.zeros((nz + 1,) + div.shape[1:], dtype=div.dtype)
    # w[k] = -dz * sum_{k'=k}^{nz-1} div[k']
    w[:nz] = -dz * np.cumsum(div[::-1], axis=0)[::-1]
    return w


def w_at_centres(w_interface: np.ndarray) -> np.ndarray:
    return 0.5 * (w_interface[:-1] + w_interface[1:])


def buoyancy_potential(b: np.ndarray, dz: float) -> np.ndarray:
    """Phi[k] = int_z^0 b dz' = dz*(0.5*b[k] + sum_{k'<k} b[k']) (spec S7.3)."""
    above = np.concatenate([np.zeros((1,) + b.shape[1:], dtype=b.dtype),
                            np.cumsum(b[:-1], axis=0)], axis=0)
    return dz * (0.5 * b + above)


def remove_depth_mean(phi: np.ndarray) -> np.ndarray:
    """Phi' = Phi - <Phi>_z (spec S7.2). Without this the barotropic mode is
    spuriously forced and the internal-wave solution stops being exact."""
    return phi - np.mean(phi, axis=0, keepdims=True)
