#########################################################################
#  Module: advection_v06                                                #
#  Description: Third-order upwind-biased face interpolation of spec    #
#               v0.6 (docs/03 S11.3): the van Leer kappa scheme with    #
#               kappa = 1/3 (MITgcm advection 33 / ROMS UP3), with a    #
#               Sweby/Superbee TVD-limited variant, on the periodic     #
#               index layout with wall fallback to first-order upwind   #
#               wherever the far stencil cell is dry. Used for tracers  #
#               (with real face thickness) and for the advected         #
#               velocity in the flux-form momentum terms.               #
#  Pipeline: model3d_v05 -> advection_v06 -> tendency                   #
#########################################################################

from __future__ import annotations

import numpy as np

KAPPA = 1.0 / 3.0


def kappa_face(c: np.ndarray, vel: np.ndarray, axis: int, wet: np.ndarray | None,
               limited: bool) -> np.ndarray:
    """Value on the face between c[i] and c[i+1] (index i) for the advecting
    velocity `vel` defined on that face; periodic rolls, upwind1 fallback
    where the far cell (i-1 for vel>0, i+2 for vel<0) is dry."""
    cm = np.roll(c, 1, axis=axis)          # c[i-1]
    cp = np.roll(c, -1, axis=axis)         # c[i+1]
    cpp = np.roll(c, -2, axis=axis)        # c[i+2]
    if limited:
        # Superbee on the local slope ratio, symmetric for either sign.
        d_c = cp - c                                       # C[i+1] - C[i]
        eps = np.where(d_c == 0.0, 1.0, d_c)
        r_pos = np.where(d_c == 0.0, 0.0, (c - cm) / eps)
        r_neg = np.where(d_c == 0.0, 0.0, (cpp - cp) / eps)
        phi_pos = np.maximum(0.0, np.maximum(np.minimum(2.0 * r_pos, 1.0), np.minimum(r_pos, 2.0)))
        phi_neg = np.maximum(0.0, np.maximum(np.minimum(2.0 * r_neg, 1.0), np.minimum(r_neg, 2.0)))
        f_pos = c + 0.5 * phi_pos * d_c
        f_neg = cp - 0.5 * phi_neg * d_c
    else:
        f_pos = c + 0.25 * ((1.0 - KAPPA) * (c - cm) + (1.0 + KAPPA) * (cp - c))
        f_neg = cp - 0.25 * ((1.0 - KAPPA) * (cpp - cp) + (1.0 + KAPPA) * (cp - c))
    if wet is not None:
        far_pos = np.roll(wet, 1, axis=axis) > 0.0
        far_neg = np.roll(wet, -2, axis=axis) > 0.0
        f_pos = np.where(far_pos, f_pos, c)
        f_neg = np.where(far_neg, f_neg, cp)
    return np.where(vel > 0.0, f_pos, f_neg)


def kappa_face_z(c: np.ndarray, w: np.ndarray, wet3: np.ndarray, limited: bool) -> np.ndarray:
    """Tracer on interior interfaces 1..nz-1 for w positive UPWARD (from
    layer k below toward layer k-1 above). Returns [nz-1, ny, nx]."""
    nz = c.shape[0]
    above, below = c[:-1], c[1:]                     # at interface k: c[k-1], c[k]
    pad = np.zeros_like(c[:1])
    far_up = np.concatenate([pad, c[:-2]], axis=0)   # c[k-2] for interface k>=2
    far_dn = np.concatenate([c[2:], pad], axis=0)    # c[k+1] for interface k<=nz-2
    has_up = np.concatenate([np.zeros_like(wet3[:1]), wet3[:-2]], axis=0) > 0.0
    has_dn = np.concatenate([wet3[2:], np.zeros_like(wet3[:1])], axis=0) > 0.0
    d = above - below
    if limited:
        eps = np.where(d == 0.0, 1.0, d)
        r_up = np.where(d == 0.0, 0.0, (below - far_dn) / eps)      # upward flow: upwind is below
        r_dn = np.where(d == 0.0, 0.0, (far_up - above) / eps)
        phi_up = np.maximum(0.0, np.maximum(np.minimum(2.0 * r_up, 1.0), np.minimum(r_up, 2.0)))
        phi_dn = np.maximum(0.0, np.maximum(np.minimum(2.0 * r_dn, 1.0), np.minimum(r_dn, 2.0)))
        f_up = below + 0.5 * phi_up * d
        f_dn = above - 0.5 * phi_dn * d
    else:
        f_up = below + 0.25 * ((1.0 - KAPPA) * (below - far_dn) + (1.0 + KAPPA) * (above - below))
        f_dn = above - 0.25 * ((1.0 - KAPPA) * (far_up - above) + (1.0 + KAPPA) * (above - below))
    f_up = np.where(has_dn, f_up, below)
    f_dn = np.where(has_up, f_dn, above)
    return np.where(w[1:nz] > 0.0, f_up, f_dn)


def momentum_advection_up3(u, v, w, dx, dy, dz, mask3u, mask3v, limited: bool):
    """Flux-form momentum advection with kappa-interpolated advected velocity
    (spec S11.3); the advecting velocities are those of S9.2."""
    nz = u.shape[0]
    # ---- u: x flux at cell centres i (between faces i-1 and i)
    ubar_c = 0.5 * (np.roll(u, 1, axis=-1) + u)                       # advecting, at centre i
    # face between u[i-1] and u[i] is index i-1 in kappa_face's convention
    u_at_c = np.roll(kappa_face(u, np.roll(ubar_c, -1, axis=-1), -1, mask3u, limited), 1, axis=-1)
    adv_ux = (np.roll(ubar_c * u_at_c, -1, axis=-1) - ubar_c * u_at_c) / dx
    vbar = 0.5 * (v + np.roll(v, -1, axis=-1))                        # corner (i+1/2, j+1/2)
    u_at_y = kappa_face(u, vbar, -2, mask3u, limited)
    fy_u = vbar * u_at_y
    adv_uy = (fy_u - np.roll(fy_u, 1, axis=-2)) / dy
    wbar_u = 0.5 * (w + np.roll(w, -1, axis=-1))
    fz_u = np.zeros_like(wbar_u)
    if nz > 1:
        fz_u[1:nz] = wbar_u[1:nz] * kappa_face_z(u, wbar_u, mask3u, limited)
    adv_uz = (fz_u[:nz] - fz_u[1:nz + 1]) / dz
    # ---- v
    vbar_c = 0.5 * (np.roll(v, 1, axis=-2) + v)
    v_at_c = np.roll(kappa_face(v, np.roll(vbar_c, -1, axis=-2), -2, mask3v, limited), 1, axis=-2)
    adv_vy = (np.roll(vbar_c * v_at_c, -1, axis=-2) - vbar_c * v_at_c) / dy
    ubar = 0.5 * (u + np.roll(u, -1, axis=-2))
    v_at_x = kappa_face(v, ubar, -1, mask3v, limited)
    fx_v = ubar * v_at_x
    adv_vx = (fx_v - np.roll(fx_v, 1, axis=-1)) / dx
    wbar_v = 0.5 * (w + np.roll(w, -1, axis=-2))
    fz_v = np.zeros_like(wbar_v)
    if nz > 1:
        fz_v[1:nz] = wbar_v[1:nz] * kappa_face_z(v, wbar_v, mask3v, limited)
    adv_vz = (fz_v[:nz] - fz_v[1:nz + 1]) / dz
    return adv_ux + adv_uy + adv_uz, adv_vx + adv_vy + adv_vz
