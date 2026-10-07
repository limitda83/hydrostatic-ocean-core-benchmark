#########################################################################
#  Module: closure                                                      #
#  Description: One-equation TKE vertical turbulence closure of spec    #
#               v0.6 (docs/03 S11.2), Gaspar et al. (1990) form: TKE   #
#               at layer interfaces, shear and buoyancy production      #
#               explicit, dissipation linearised implicit, vertical     #
#               diffusion of TKE fully implicit (one tridiagonal per    #
#               column), mixing length limited by stratification and    #
#               by the distance to both boundaries. Returns the new TKE #
#               and the interface viscosity / diffusivity that the      #
#               stepper then feeds into the S7.4 tridiagonals and the   #
#               S10.5 Helmholtz coefficients, rebuilt every step.       #
#  Pipeline: model3d_v05 (start of step) -> closure -> vertical_var     #
#########################################################################

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from libs.core.vertical import thomas_batched


@dataclass(frozen=True)
class TKEParams:
    c_k: float = 0.1
    c_eps: float = 0.7
    pr_t: float = 1.0
    kappa_vk: float = 0.4
    z_0: float = 0.1
    e_min: float = 1.0e-6
    n2_min: float = 1.0e-8
    l_min: float = 0.01
    e_bb: float = 3.75
    mxl: str = "integral"      # "integral" (O(nz^2), S11.2) | "recursive" (O(nz))


def _safe(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0.0, 1.0 / np.where(x > 0.0, x, 1.0), 0.0)


def interface_geometry(dz3: np.ndarray, mask3: np.ndarray):
    """Depth of each interface below the surface, distance to the wet bottom,
    the interface wet mask and the centre-to-centre spacing across it.

    Interface k (0..nz) lies between layer k-1 above and layer k below.
    Interface 0 is the surface, interface nz the bottom of the last layer.
    An interface is 'interior wet' when both adjacent layers are wet.
    """
    nz = dz3.shape[0]
    zi = np.concatenate([np.zeros((1,) + dz3.shape[1:]), np.cumsum(dz3, axis=0)], axis=0)
    h = zi[-1]                                       # wet column depth
    z_s = zi                                         # depth below surface
    z_b = np.maximum(h[None] - zi, 0.0)              # distance to the bottom
    wet = np.zeros_like(zi)
    wet[1:nz] = mask3[:-1] * mask3[1:]
    dzi = np.empty_like(zi)
    dzi[0] = dz3[0]
    dzi[1:nz] = 0.5 * (dz3[:-1] + dz3[1:])
    dzi[nz] = dz3[-1]
    return z_s, z_b, wet, dzi


def gaspar_lengths(e, n2, dzi, z_s, z_b, wet, p: TKEParams):
    """Discrete Gaspar (1990) lengths on the interface grid: the distance a
    parcel with kinetic energy e can travel against the stratification,
    int_0^l N^2 s ds = e (uniform N: l = sqrt(2e)/N). Marching away from
    interface k one interface at a time, crossing interface k-+j costs
    N^2_j dzi_j (d + dzi_j/2) with d the distance already travelled; the last
    step is truncated where the budget runs out. Bounded by the wall distances."""
    nz1 = e.shape[0]
    n2p = np.maximum(n2, 0.0)
    e_k = np.maximum(e, p.e_min)
    l_up = np.zeros_like(e)
    l_dn = np.zeros_like(e)
    b_up = e_k.copy()
    b_dn = e_k.copy()
    for j in range(1, nz1):
        for direction in ("up", "dn"):
            if direction == "up":
                n2_j = np.concatenate([np.zeros((j,) + e.shape[1:]), n2p[:-j]], axis=0)
                dz_j = np.concatenate([np.zeros((j,) + e.shape[1:]), dzi[:-j]], axis=0)
                l, budget = l_up, b_up
            else:
                n2_j = np.concatenate([n2p[j:], np.zeros((j,) + e.shape[1:])], axis=0)
                dz_j = np.concatenate([dzi[j:], np.zeros((j,) + e.shape[1:])], axis=0)
                l, budget = l_dn, b_dn
            cost = n2_j * dz_j * (l + 0.5 * dz_j)
            full = cost <= budget
            # partial step: solve N^2 (l s + s^2/2) = budget for s in (0, dz_j)
            a = np.where(n2_j > 0.0, n2_j, 1.0)
            s_part = np.sqrt(np.maximum(l * l + 2.0 * budget / a, 0.0)) - l
            step = np.where(full, dz_j, np.where(n2_j > 0.0, np.clip(s_part, 0.0, dz_j), dz_j))
            l_new = l + np.where(budget > 0.0, step, 0.0)
            budget_new = np.where(full, budget - cost, 0.0)
            if direction == "up":
                l_up, b_up = l_new, budget_new
            else:
                l_dn, b_dn = l_new, budget_new
    l_up = np.minimum(l_up, z_s + p.z_0) * wet
    l_dn = np.minimum(l_dn, z_b + p.z_0) * wet
    return l_up, l_dn


def recursive_lengths(e, n2, dzi, z_s, z_b, wet, p: TKEParams):
    """Blanke & Delecluse (1993) / NEMO nn_mxl=2 mixing length: the buoyancy
    length sqrt(2e/N^2) capped by the distance to each wall, then limited to
    grow no faster than the grid by one downward and one upward sweep. Two
    O(nz) passes instead of the O(nz^2) energy budget of gaspar_lengths, so
    the two are one axis of the spec (S11.2, `mxl`), not two algorithms.

    Returns (l_up, l_dn) UNFLOORED, exactly like gaspar_lengths: the caller
    applies max(l_min, .) when it forms l_k and l_eps. Both branches of the
    axis must leave the floor to the caller, or they would differ by more than
    the algorithm under test. This form has no up/down asymmetry, so it returns
    the same array twice and l_k = l_eps follows.
    """
    nz1 = e.shape[0]
    n2c = np.maximum(n2, p.n2_min)
    e_k = np.maximum(e, p.e_min)
    l = np.sqrt(2.0 * e_k / n2c)
    l = np.minimum(l, np.minimum(z_s + p.z_0, z_b + p.z_0))
    for k in range(1, nz1):                      # downward: limit by the one above
        l[k] = np.minimum(l[k], l[k - 1] + dzi[k])
    for k in range(nz1 - 2, -1, -1):             # upward: limit by the one below
        l[k] = np.minimum(l[k], l[k + 1] + dzi[k + 1])
    l = l * wet
    return l, l


def mixing_lengths(e, n2, dzi, z_s, z_b, wet, p: TKEParams):
    """Dispatch on the `mxl` axis of spec S11.2."""
    if p.mxl == "recursive":
        return recursive_lengths(e, n2, dzi, z_s, z_b, wet, p)
    if p.mxl != "integral":
        raise ValueError(f"unknown mixing length form {p.mxl!r} (integral|recursive)")
    return gaspar_lengths(e, n2, dzi, z_s, z_b, wet, p)


def tke_step(e: np.ndarray, u: np.ndarray, v: np.ndarray, b: np.ndarray, dz3: np.ndarray,
             mask3: np.ndarray, ustar2: np.ndarray, dt: float, p: TKEParams,
             nu_b: float, kappa_b: float, K_m_old: np.ndarray, K_h_old: np.ndarray):
    """One TKE step. All 3D arrays are [k, j, i]; e, K are on interfaces [nz+1].

    Returns (e_new, K_m, K_h) with K_m, K_h on interfaces, background added.
    """
    nz = dz3.shape[0]
    z_s, z_b, wet, dzi = interface_geometry(dz3, mask3)
    inv_dzi = _safe(dzi)
    # cell-centre velocities (u, v live on faces), then the interface shear
    uc = 0.5 * (u + np.roll(u, 1, axis=-1))
    vc = 0.5 * (v + np.roll(v, 1, axis=-2))
    du = np.zeros_like(e)
    dv = np.zeros_like(e)
    db = np.zeros_like(e)
    du[1:nz] = (uc[:-1] - uc[1:]) * inv_dzi[1:nz]
    dv[1:nz] = (vc[:-1] - vc[1:]) * inv_dzi[1:nz]
    db[1:nz] = (b[:-1] - b[1:]) * inv_dzi[1:nz]         # N^2 = db/dz, z upward
    shear2 = (du ** 2 + dv ** 2) * wet
    n2 = db * wet

    # Gaspar (1990) mixing lengths from the CURRENT TKE: l_up (l_dn) is the
    # distance a parcel of energy e can rise (sink) against the stratification,
    # int N^2 dz <= e, bounded by the distance to the surface (bottom).
    l_up, l_dn = mixing_lengths(e, n2, dzi, z_s, z_b, wet, p)
    l_k = np.maximum(p.l_min, np.sqrt(l_up * l_dn))
    l_eps = np.maximum(p.l_min, np.minimum(l_up, l_dn))
    length = l_k

    prod = (K_m_old - nu_b) * shear2 - (K_h_old - kappa_b) * n2
    diss_coef = p.c_eps * np.sqrt(np.maximum(e, p.e_min)) / l_eps
    K_e = K_m_old

    # implicit vertical diffusion of e on the interface grid: unknowns are the
    # interior interfaces 1..nz-1; the surface (Dirichlet) and the bottom
    # (e_min) enter as boundary values. Row k couples k-1 and k+1 through
    # K_e at the layer centres (average of the two interface values) over
    # the layer thickness dz3.
    K_lay = 0.5 * (K_e[:-1] + K_e[1:])                 # [nz] at layer centres
    inv_dz3 = _safe(dz3)
    a_up = np.zeros_like(e)                              # coefficient to k-1
    a_dn = np.zeros_like(e)                              # coefficient to k+1
    a_up[1:nz] = dt * K_lay[:-1] * inv_dz3[:-1] * inv_dzi[1:nz]
    a_dn[1:nz] = dt * K_lay[1:] * inv_dz3[1:] * inv_dzi[1:nz]
    sub = -a_up
    sup = -a_dn
    diag = 1.0 + a_up + a_dn + dt * diss_coef
    rhs = e + dt * prod
    # boundary interfaces and dry interfaces: identity rows with their values
    e_surf = p.e_bb * ustar2
    bnd = wet <= 0.0
    diag = np.where(bnd, 1.0, diag)
    sub = np.where(bnd, 0.0, sub)
    sup = np.where(bnd, 0.0, sup)
    rhs = np.where(bnd, p.e_min, rhs)
    rhs[0] = np.where(mask3[0] > 0.0, e_surf, p.e_min)
    e_new = thomas_batched(sub, diag, sup, rhs)
    e_new = np.maximum(e_new, p.e_min)
    e_new = np.where(wet > 0.0, e_new, p.e_min)
    e_new[0] = rhs[0]                                    # surface Dirichlet value
    # eddy coefficients from the NEW TKE and the same mixing length
    K_m = p.c_k * length * np.sqrt(e_new) * wet + nu_b
    K_h = (K_m - nu_b) / p.pr_t + kappa_b
    return e_new, K_m, K_h
